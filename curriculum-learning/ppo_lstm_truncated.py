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


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )  

        self.linear = nn.Sequential(
            nn.Linear(2304, 512),  # 64, 6, 6
            nn.ReLU()
        )

        self.lstm = nn.LSTM(512, 256, 1)

        self.actor_head = nn.Linear(256, 7)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x, hidden):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)
        x = x.unsqueeze(0)

        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)

        actor = F.softmax(self.actor_head(x), dim=-1)
        critic = self.critic_head(x)

        return actor, critic, hidden
    
    def loss_forward(self, x, dones, truncated_steps):
        n_envs, n_steps = x.size(0), x.size(1)

        x = x.view(-1, 7, 10, 10)

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)
        x = x.view(n_envs, n_steps, -1)


        done_steps = (dones == True).any(0).nonzero().squeeze(1).tolist()
        done_steps = [-1] + done_steps + ([] if done_steps[-1] == (n_steps - 1) else [n_steps-1])

        hidden = torch.zeros((1, n_envs, 256), device=device)
        cell = torch.zeros((1, n_envs, 256), device=device)
        done = torch.full((n_envs, ), True,  device=device)
        rnn_outputs = []

        for i in range(len(done_steps) - 1):
            start_step = done_steps[i] + 1
            end_step = done_steps[i+1]

            hidden[:, done] = 0
            cell[:, done] = 0
            done = dones[:, end_step]

            for truncated_start_step in range(start_step,  end_step+1, truncated_steps):
                hidden, cell = hidden.detach().requires_grad_(), cell.detach().requires_grad_()

                rnn_input = x[:, truncated_start_step: min(truncated_start_step+truncated_steps, end_step+1)].permute(1, 0, 2)
                rnn_output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
                rnn_outputs.append(rnn_output.permute(1, 0, 2))

        # assert sum([x.size(1) for x in rnn_outputs]) == n_steps

        x = torch.cat(rnn_outputs, dim=1)
        x = x.view(n_envs * n_steps, -1)

        actor = F.softmax(self.actor_head(x), dim=-1)
        critic = self.critic_head(x)

        return actor, critic


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

    def sample(self, model, steps, gamma, lamda):
        states = torch.zeros((self.n_workers, steps, 7, 10, 10), dtype=torch.float32, device=device)
        values = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        actions = torch.zeros((self.n_workers, steps), dtype=torch.uint8, device=device)
        log_probabilities = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        rewards = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        dones = torch.zeros((self.n_workers, steps), dtype=torch.bool, device=device)

        advantages = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        observation = torch.tensor(self.reset(), device=device)
        hidden = torch.zeros((1, self.n_workers, 256), device=device)
        cell = torch.zeros((1, self.n_workers, 256), device=device)

        for t in range(steps):
            with torch.no_grad():
                states[:, t] = observation

                pi, v, (hidden, cell) = model(observation, (hidden, cell))
                values[:, t] = v.squeeze(1)

                p = Categorical(pi)
                action = p.sample()
                actions[:, t] = action
                log_probabilities[:, t] = p.log_prob(action)

            observation, reward, done, info = self.step(action.tolist())
            observation = torch.tensor(observation, device=device)
            rewards[:, t] = torch.tensor(reward, device=device)
            dones[:, t] = torch.tensor(done, device=device)
            hidden[:, done] = 0
            cell[:, done] = 0

        _, last_value, _ = model(observation, (hidden, cell))
        last_value = last_value.detach().squeeze(1)
        last_advantage = 0

        for t in reversed(range(steps)):
            mask = 1.0 - dones[:, t].int()

            delta = rewards[:, t] + gamma * last_value * mask - values[:, t]

            last_advantage = delta + gamma * lamda * last_advantage * mask

            advantages[:, t] = last_advantage

            last_value = values[:, t]

        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        targets = advantages + values

        total_reward = rewards.sum().item()

        return states, values, actions, log_probabilities, advantages, normalized_advantages, targets, dones, total_reward

def compute_loss(model, states, values, actions, log_probabilities, advantages, normalized_advantages, targets, dones, truncated_steps, clip_range,  value_coefficient, entropy_coefficient):
    values = values.view(-1)
    actions = actions.view(-1)
    log_probabilities = log_probabilities.view(-1)
    normalized_advantages = normalized_advantages.view(-1)
    targets = targets.view(-1)

    pi, v = model.loss_forward(states, dones, truncated_steps)
    v = v.squeeze(1)

    p = Categorical(pi)
    log_actions = p.log_prob(actions)

    ratio = torch.exp(log_actions - log_probabilities)
    surrogate1 = ratio * normalized_advantages
    surrogate2 = ratio.clamp(1-clip_range, 1+clip_range) * normalized_advantages
    policy_loss = torch.min(surrogate1, surrogate2).mean()

    clipped_values = values + (v - values).clamp(-clip_range, clip_range)
    value_loss1 = (v - targets).square()
    value_loss2 = (clipped_values - targets).square()
    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

    entropy = p.entropy().mean()

    loss = -(policy_loss - value_coefficient * value_loss + entropy_coefficient * entropy)

    return loss

def train():
    learning_rate = 1e-4
    gamma = 0.99
    lamda = 0.95
    clip_range = 0.1
    value_coefficient = 0.5
    entropy_coefficient = 0.01
    max_grad_norm = 0.5
    truncated_steps = 20

    total_steps = 1e8  # number of timesteps
    n_envs = 32  # number of environment copies simulated in parallel
    n_sample_steps = 128  # number of steps of the environment per sample
    n_mini_batches = 8  # number of training minibatches per update 
                                     # For recurrent policies, should be smaller or equal than number of environments run in parallel.
    n_epochs = 4   # number of training epochs per update
    batch_size = n_envs * n_sample_steps
    n_envs_per_batch = n_envs // n_mini_batches
    n_updates = math.ceil(total_steps / batch_size)
    assert (n_envs % n_mini_batches == 0)

    save_path = "./data"
    [os.makedirs(f"{save_path}/{dir}") for dir in ["data", "model", "plot", "runs"] if not os.path.exists(f"{save_path}/{dir}")]

    envs = ParallelEnv(n_envs)
    model  = PPO().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    log = pd.DataFrame([], columns=["time", "update", "step", "reward", "average_reward"])
    writer = SummaryWriter()

    for update in range(1, n_updates+1):

        states, values, actions, log_probabilities, advantages, normalized_advantages, targets, dones, total_reward = envs.sample(model, n_sample_steps, gamma, lamda)

        for _ in range(n_epochs):

            indexes = torch.randperm(n_envs)

            for i in range(0, n_envs, n_envs_per_batch):
                mini_batch_indexes = indexes[i: i + n_envs_per_batch]

                loss = compute_loss(
                    model, 
                    states[mini_batch_indexes], values[mini_batch_indexes], 
                    actions[mini_batch_indexes], log_probabilities[mini_batch_indexes], 
                    advantages[mini_batch_indexes], normalized_advantages[mini_batch_indexes], targets[mini_batch_indexes],
                    dones[mini_batch_indexes],
                    truncated_steps, clip_range, value_coefficient, entropy_coefficient
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

