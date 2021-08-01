import math
from datetime import datetime
import os
from distutils.dir_util import copy_tree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

import boxoban_level_collection as levelCollection
from boxoban_environment import BoxobanEnvironment


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DilatedLSTM(nn.Module):
    def __init__(self,  input_size, hidden_size,  dilation=10):
        super(DilatedLSTM, self).__init__()

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.dilation = dilation

    """
    input: batch×input_size
    states: (cursor, hidden, cell)
    cursor: int
    hx: batch×dilation×hidden_size
    cx: batch×dilation×hidden_size
    """
    def forward(self, inputs, states):
        cursor, hidden, cell = states

        hx, cx = hidden.clone(), cell.clone()

        hx[:, cursor], cx[:, cursor] = self.lstm(inputs, (hidden[:, cursor], cell[:, cursor]))

        mask = torch.full((self.dilation, ), True,  device=device)
        mask[cursor] = False
        remaining_hx = hx[:, mask]

        x = (hx[:, cursor] + remaining_hx.sum(dim=1).detach()) / self.dilation

        cursor = (cursor + 1) % self.dilation

        return x, (cursor, hx, cx)

class Perception(nn.Module):
    def __init__(self):
        super(Perception, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(2304, 256),  # 64, 6, 6
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x

class Worker(nn.Module):
    def __init__(self,):
        super(Worker, self).__init__()

        self.Wrnn = nn.LSTMCell(256, 16 * 8)
        self.phi = nn.Linear(256, 16, bias=False)
        self.critic_head = nn.Linear(16 * 8, 1)

    """
    z: batch×d
    hidden: (hidden, cell)  batch×(k×actions)
    goals:  list of length C+1, batch×d
    """
    def forward(self, z, hidden, goals):
        hx, _ = hidden = self.Wrnn(z, hidden)

        u = hx.view(hx.size(0), 8, 16)

        goals = torch.stack(goals).sum(dim=0).detach()
        w = self.phi(goals).unsqueeze(-1)

        actor = torch.bmm(u, w).squeeze(-1)
        actor = F.softmax(actor, dim=-1)

        critic = self.critic_head(hx)

        return actor, critic, hidden

    """
    states: list of length C+1, batch×d
    goals: list of length C+1, batch×d
    masks: list of length C+1, batch
    """
    def intrinsic_reward(self, states, goals, masks):
        batch_size = masks[0].size(0)

        r_i = torch.zeros(batch_size, device=device)
        mask = torch.ones(batch_size, device=device)

        for i in range(1, 11):
            mask = mask * masks[10-i]

            r_i_t = F.cosine_similarity(states[10] - states[10-i], goals[10-i])

            r_i += (mask * r_i_t)

        return r_i.detach() / 10

class Manager(nn.Module):
    def __init__(self):
        super(Manager, self).__init__()

        self.Mspace = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.Mrnn = DilatedLSTM(256, 256, 10)
        self.critic_head = nn.Linear(256, 1)

    """
    z: batch×d
    hidden: (cursor, hidden, cell)  batch×dilation×d
    """
    def forward(self, z, hidden):
        state = self.Mspace(z)

        goal_hat, hidden = self.Mrnn(state, hidden)
        goal = F.normalize(goal_hat)

        if (0.05 > torch.rand(1).item()):
            # To encourage exploration in transition policy,
            # at every step with a small probability ε
            # we emit a random goal sampled from a uni-variate Gaussian.
            goal = torch.randn_like(goal, requires_grad=False)

        critic = self.critic_head(goal_hat)

        return critic, hidden, state, goal

    """
    states: list of length C+1, batch×d
    goals: list of length C+1, batch×d
    masks: list of length C+1, batch
    """
    def state_goal_similarity(self, states, goals, masks):
        mask = torch.stack(masks[0:10]).prod(dim=0)
        cosine_similarity = F.cosine_similarity(states[10] - states[0], goals[0])
        cosine_similarity = mask * cosine_similarity

        return cosine_similarity

class FeudalNetwork(nn.Module):
    def __init__(self):
        super(FeudalNetwork, self).__init__()

        self.perception = Perception()
        self.manager = Manager()
        self.worker = Worker()

    """
    x: batch×d
    manager_hidden: (cursor, hidden, cell)  batch×dilation×d
    worker_hidden: (hidden, cell)  batch×(k×actions)
    states: list of length C+1, batch×d
    goals: list of length C+1, batch×d
    mask: batch
    """
    def forward(self, x, manager_hidden, worker_hidden, states, goals, mask):
        z = self.perception(x)

        manager_value, manager_hidden, state, goal = self.manager(z, manager_hidden)

        goals = [goal * mask.unsqueeze(-1) for goal in goals]

        goals.pop(0)
        states.pop(0)
        goals.append(goal)
        states.append(state.detach())  # states never have gradients active

        worker_pi, worker_value, worker_hidden = self.worker(z, worker_hidden, goals)

        return manager_value, manager_hidden, worker_pi, worker_value, worker_hidden, states, goals

    def state_goal_similarity(self, states, goals, masks):
        return self.manager.state_goal_similarity(states, goals, masks)

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

def collection_worker(queue):
    while True:
        queue.put(levelCollection.random())

def worker(master, collection):
    while True:
        cmd, data = master.recv()
        if cmd == 'step':
            observation, reward, done, info = env.step(data)
            if done:
                (id, score, trajectory, room, topology) = collection.get()
                env = BoxobanEnvironment(room, topology)
                observation = env.observation
            master.send((observation, reward, done, info))
        elif cmd == 'reset':
            (id, score, trajectory, room, topology) = collection.get()
            env = BoxobanEnvironment(room, topology)
            master.send(env.observation)
        elif cmd == 'close':
            master.close()
            collection.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.workers = []

        self.master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(n_workers)])

        queue = mp.Queue(n_workers)

        for worker_end in worker_ends:
            p = mp.Process(target=worker, args=(worker_end, queue))
            p.daemon = True
            p.start()
            self.workers.append(p)

        p = mp.Process(target=collection_worker, args=(queue, ))
        p.daemon = True
        p.start()
        self.collection = p

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
            
        results = [master_end.recv() for master_end in self.master_ends]
        observations, rewards, dones, infos = zip(*results)

        return np.stack(observations), np.stack(rewards), np.stack(dones), infos

    def close(self):
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()

    def sample(self, model, steps, manager_gamma, worker_gamma, manager_lamda, worker_lamda, alpha):
        manager_values = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)
        worker_values = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        log_probabilities = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)
        entropies  = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        rewards = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)
        intrinsic_rewards = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        masks = torch.zeros((self.n_workers, steps), dtype=torch.int32, device=device)

        state_goal_similarities = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        manager_advantages = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)
        worker_advantages = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        window_states = [torch.zeros(self.n_workers, 256, device=device) for _ in range(11)]
        window_goals = [torch.zeros(self.n_workers, 256, device=device) for _ in range(11)]
        window_masks = [torch.zeros(self.n_workers, dtype=torch.int32, device=device) for _ in range(11)]

        manager_hidden = (
            0,
            torch.zeros(self.n_workers, 10, 256, device=device),
            torch.zeros(self.n_workers, 10, 256, device=device),
        )
        worker_hidden = (
            torch.zeros(self.n_workers, 128, device=device),
            torch.zeros(self.n_workers, 128, device=device),
        )
        observation = torch.tensor(self.reset(), device=device)

        for t in range(steps):
            manager_value, (cursor, manager_hx, manager_cx), worker_pi, worker_value, (worker_hx, worker_cx), window_states, window_goals = model(observation, manager_hidden, worker_hidden, window_states, window_goals, window_masks[-1])

            manager_values[:, t] = manager_value.squeeze(1)
            worker_values[:, t] = worker_value.squeeze(1)

            p = Categorical(worker_pi)
            action = p.sample()
            log_probabilities[:, t] = p.log_prob(action)
            entropies[:, t] = p.entropy()

            observation, reward, done, info = self.step(action.tolist())
            observation = torch.tensor(observation, device=device)

            rewards[:, t] = torch.tensor(reward, device=device)

            mask = 1 - torch.tensor(done, device=device).int()
            masks[:, t] = mask
            window_masks.pop(0)
            window_masks.append(mask)

            manager_hidden = (
                cursor, 
                manager_hx * mask.unsqueeze(-1).unsqueeze(-1), 
                manager_cx * mask.unsqueeze(-1).unsqueeze(-1)
            )

            worker_hidden = (
                worker_hx * mask.unsqueeze(-1), 
                worker_cx * mask.unsqueeze(-1)
            )

            if steps >= 1:
                intrinsic_rewards[:, t-1] = model.intrinsic_reward(window_states, window_goals, window_masks)
           
            if steps >= 10:
                state_goal_similarities[:, t-10] =  model.state_goal_similarity(window_states, window_goals, window_masks)


        with torch.no_grad():
            manager_last_value, _, _, worker_last_value, _, window_states, window_goals = model(observation, manager_hidden, worker_hidden, window_states, window_goals, window_masks[-1])
            intrinsic_rewards[:, -1] = model.intrinsic_reward(window_states, window_goals, window_masks)
            manager_last_value = manager_last_value.squeeze(1)
            worker_last_value = worker_last_value.squeeze(1)
            manager_last_advantage = 0
            worker_last_advantage = 0

        for t in reversed(range(steps)):
            manager_delta = rewards[:, t] + manager_gamma * manager_last_value * masks[:, t] - manager_values[:, t]
            manager_last_advantage = manager_delta + manager_gamma * manager_lamda * manager_last_advantage * masks[:, t]
            manager_advantages[:, t] = manager_last_advantage
            manager_last_value = manager_values[:, t]

            worker_delta = rewards[:, t] + alpha * intrinsic_rewards[:, t] + worker_gamma * worker_last_value * masks[:, t] - worker_values[:, t]
            worker_last_advantage = worker_delta + worker_gamma * worker_lamda * worker_last_advantage * masks[:, t]
            worker_advantages[:, t] = worker_last_advantage
            worker_last_value = worker_values[:, t]

        state_goal_similarities = state_goal_similarities.view(-1)
        log_probabilities = log_probabilities.view(-1)
        entropies = entropies.view(-1)

        manager_advantages = manager_advantages.view(-1)
        worker_advantages = worker_advantages.view(-1)
        manager_normalized_advantages = (manager_advantages - manager_advantages.mean()) / (manager_advantages.std() + 1e-8)
        worker_normalized_advantages = (worker_advantages - worker_advantages.mean()) / (worker_advantages.std() + 1e-8)

        total_reward = rewards.sum().item()

        return state_goal_similarities, log_probabilities,  manager_advantages, worker_advantages, manager_normalized_advantages, worker_normalized_advantages, entropies, total_reward

def compute_loss(state_goal_similarities, log_probabilities, manager_advantages, worker_advantages, manager_normalized_advantages, worker_normalized_advantages, entropies,  manager_value_coefficient, worker_value_coefficient, entropy_coefficient):
    manager_policy_loss = (state_goal_similarities * manager_normalized_advantages.detach()).mean()
    worker_policy_loss = (log_probabilities * worker_normalized_advantages.detach()).mean()
    
    manager_value_loss = 0.5 * manager_advantages.square().mean()
    worker_value_loss = 0.5 * worker_advantages.square().mean()

    entropy = entropies.mean()

    loss = - (manager_policy_loss + worker_policy_loss - manager_value_coefficient * manager_value_loss - worker_value_coefficient * worker_value_loss + entropy_coefficient * entropy)

    return loss

def train():
    learning_rate = 3e-4
    manager_gamma = 0.99
    worker_gamma = 0.95
    manager_lamda = 0.95
    worker_lamda = 0.95
    alpha = 0.9
    manager_value_coefficient = 0.5
    worker_value_coefficient = 0.5
    entropy_coefficient = 0.01
    max_grad_norm = 0.5 

    total_steps = 1e8  # number of timesteps
    n_envs = 32  # number of environment copies simulated in parallel
    n_sample_steps = 128  # number of steps of the environment per sample
    n_mini_batches = 1  # number of training minibatches per update 
                                     # For recurrent policies, should be smaller or equal than number of environments run in parallel.
    batch_size = n_envs * n_sample_steps
    mini_batch_size = batch_size // n_mini_batches
    n_updates = math.ceil(total_steps / batch_size)
    assert (batch_size % n_mini_batches == 0)

    save_path = "./data"
    [os.makedirs(f"{save_path}/{dir}") for dir in ["data", "model", "plot", "runs"] if not os.path.exists(f"{save_path}/{dir}")]

    envs = ParallelEnv(n_envs)
    model  = FeudalNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    log = pd.DataFrame([], columns=["time", "update", "step", "reward", "average_reward"])
    writer = SummaryWriter()

    for update in range(1, n_updates+1):
        state_goal_similarities, log_probabilities,  manager_advantages, worker_advantages, manager_normalized_advantages, worker_normalized_advantages, entropies, total_reward = envs.sample(model, n_sample_steps, manager_gamma, worker_gamma, manager_lamda, worker_lamda, alpha)

        indexes = torch.randperm(batch_size)

        for i in range(0, batch_size, mini_batch_size):
            mini_batch_indexes = indexes[i: i + mini_batch_size]

            loss = compute_loss(
                state_goal_similarities[mini_batch_indexes], log_probabilities[mini_batch_indexes],
                manager_advantages[mini_batch_indexes], worker_advantages[mini_batch_indexes],
                manager_normalized_advantages[mini_batch_indexes], worker_normalized_advantages[mini_batch_indexes],
                entropies[mini_batch_indexes],
                manager_value_coefficient, worker_value_coefficient, entropy_coefficient
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

        step += batch_size
        reward = total_reward/n_envs
        tail_rewards = log["reward"].tail(99)
        average_reward = (tail_rewards.sum() + reward) / (tail_rewards.count() +1)
        log.at[update] = [datetime.now(), update, step, reward, average_reward]
        
        writer.add_scalar('Reward', reward, update)
        writer.add_scalar('Average reward', average_reward, update)

        print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {update},{step}: {reward:.2f}")

        if update % 122 == 0:
            fig = log["average_reward"].plot().get_figure()
            fig.savefig(f"{save_path}/plot/{step}.png")
            copy_tree("./runs", f"{save_path}/runs")

            torch.save(model.state_dict(), f"{save_path}/model/{step}.pkl")
            log.to_csv(f"{save_path}/data/{step}.csv", index=False, header=True)


    fig = log["average_reward"].plot().get_figure()
    fig.savefig(f"{save_path}/plot/{step}.png")
    copy_tree("./runs", f"{save_path}/runs")
    torch.save(model.state_dict(), f"{save_path}/model/{step}.pkl")
    log.to_csv(f"{save_path}/data/{step}.csv", index=False, header=True)

    envs.close()

if __name__ == '__main__':
    train()
