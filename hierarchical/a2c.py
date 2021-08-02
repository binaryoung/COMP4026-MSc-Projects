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


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()

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

        self.actor_head = nn.Linear(256, 8)
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
        values = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        log_probabilities = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)
        entropies  = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        rewards = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        dones = torch.zeros((self.n_workers, steps), dtype=torch.bool, device=device)

        advantages = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        observation = torch.tensor(self.reset(), device=device)
        hidden = torch.zeros((1, self.n_workers, 256), device=device)
        cell = torch.zeros((1, self.n_workers, 256), device=device)

        for t in range(steps):
            pi, v, (hidden, cell) = model(observation, (hidden, cell))
            values[:, t] = v.squeeze(1)

            p = Categorical(pi)
            action = p.sample()
            log_probabilities[:, t] = p.log_prob(action)
            entropies[:, t] = p.entropy()

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

        log_probabilities = log_probabilities.view(-1)
        entropies = entropies.view(-1)

        advantages = advantages.view(-1)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_reward = rewards.sum().item()

        return log_probabilities, advantages, normalized_advantages, entropies, total_reward

def compute_loss(log_probabilities, advantages, normalized_advantages, entropies, value_coefficient, entropy_coefficient):
    policy_loss = (log_probabilities * normalized_advantages.detach()).mean()
    
    value_loss = advantages.square().mean()

    entropy = entropies.mean()

    loss = - (policy_loss - value_coefficient * value_loss + entropy_coefficient * entropy)

    return loss

def train():
    learning_rate = 7e-4
    gamma = 0.99
    lamda = 0.95
    value_coefficient = 0.5
    entropy_coefficient = 0.01
    max_grad_norm = 0.5

    total_steps = 1e8  # number of timesteps
    n_envs = 32  # number of environment copies simulated in parallel
    n_sample_steps = 128  # number of steps of the environment per sample
    batch_size = n_envs * n_sample_steps
    n_updates = math.ceil(total_steps / batch_size)

    save_path = "./data"
    [os.makedirs(f"{save_path}/{dir}") for dir in ["data", "model", "plot", "runs"] if not os.path.exists(f"{save_path}/{dir}")]

    envs = ParallelEnv(n_envs)
    model  = A2C().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    log = pd.DataFrame([], columns=["time", "update", "step", "reward", "average_reward"])
    writer = SummaryWriter()

    for update in range(1, n_updates+1):

        log_probabilities, advantages, normalized_advantages, entropies, total_reward = envs.sample(model, n_sample_steps, gamma, lamda)

        loss = compute_loss(
            log_probabilities, 
            advantages, normalized_advantages,
            entropies,
            value_coefficient, entropy_coefficient
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

