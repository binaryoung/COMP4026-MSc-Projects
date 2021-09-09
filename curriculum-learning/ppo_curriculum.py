import math
from datetime import datetime
import os
from distutils.dir_util import copy_tree
import random
from collections import deque

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


# Model
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
            nn.Linear(2304, 256),  # 64, 6, 6
            nn.ReLU()
        )

        self.actor_head = nn.Linear(256, 8)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        actor = F.softmax(self.actor_head(x), dim=-1)
        critic = self.critic_head(x)

        return actor, critic

# Level collection process
def collection_worker(queue, curriculum_cursor, curriculum_span):
    while True:
        cursor = curriculum_cursor.value
        revise_range = (16, max(cursor, 16+curriculum_span))
        learn_range = (cursor, min(cursor+curriculum_span, 250))
        range = random.choices([revise_range, learn_range], k=1, weights=[0.2, 0.8])[0]
        queue.put(levelCollection.range(*range))

# Worker process
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

# Parallel environments
class ParallelEnv:
    def __init__(self, n_workers, curriculum_cursor, curriculum_span):
        self.n_workers = n_workers
        self.workers = []

        self.master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(n_workers)])

        queue = mp.Queue(n_workers)

        for worker_end in worker_ends:
            p = mp.Process(target=worker, args=(worker_end, queue))
            p.daemon = True
            p.start()
            self.workers.append(p)

        p = mp.Process(target=collection_worker, args=(queue, curriculum_cursor, curriculum_span))
        p.daemon = True
        p.start()
        self.collection = p

    # Reset environments
    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    # Step in environments
    def step(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
            
        results = [master_end.recv() for master_end in self.master_ends]
        observations, rewards, dones, infos = zip(*results)

        return np.stack(observations), np.stack(rewards), np.stack(dones), infos

    # Close environments
    def close(self):
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()

    # Sample trajectories in environments
    def sample(self, model, steps, gamma, lamda):
        states = torch.zeros((self.n_workers, steps, 7, 10, 10), dtype=torch.float32, device=device)
        values = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        actions = torch.zeros((self.n_workers, steps), dtype=torch.uint8, device=device)
        log_probabilities = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        rewards = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        dones = torch.zeros((self.n_workers, steps), dtype=torch.bool, device=device)

        advantages = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

        observation = torch.tensor(self.reset(), device=device)

        for t in range(steps):
            with torch.no_grad():
                states[:, t] = observation
                pi, v = model(observation)
                values[:, t] = v.squeeze(1)

                p = Categorical(pi)
                action = p.sample()
                actions[:, t] = action
                log_probabilities[:, t] = p.log_prob(action)

            observation, reward, done, info = self.step(action.tolist())
            observation = torch.tensor(observation, device=device)
            rewards[:, t] = torch.tensor(reward, device=device)
            dones[:, t] = torch.tensor(done, device=device)

        _, last_value = model(observation)
        last_value = last_value.detach().squeeze(1)
        last_advantage = 0

        # Compute GAE
        for t in reversed(range(steps)):
            mask = 1.0 - dones[:, t].int()

            delta = rewards[:, t] + gamma * last_value * mask - values[:, t]

            last_advantage = delta + gamma * lamda * last_advantage * mask

            advantages[:, t] = last_advantage

            last_value = values[:, t]

        # Flatten
        states = states.view(-1, 7, 10, 10)
        values = values.view(-1)
        actions = actions.view(-1)
        log_probabilities = log_probabilities.view(-1)
        advantages = advantages.view(-1)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        targets = advantages + values

        total_reward = rewards.sum().item()

        return states, values, actions, log_probabilities, advantages, normalized_advantages, targets, total_reward

# Compute loss
def compute_loss(model, states, values, actions, log_probabilities, advantages, normalized_advantages, targets, clip_range,  value_coefficient, entropy_coefficient):
    pi, v = model(states)
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
    learning_rate = 3e-4 # learning rate
    gamma = 0.99  # gamma
    lamda = 0.95  # GAE lambda
    clip_range = 0.1 # surrogate objective clip range
    value_coefficient = 0.5  # value coefficient in loss function
    entropy_coefficient = 0.01  # entropy coefficient in loss function
    max_grad_norm = 0.5  # max gradient norm

    total_steps = 1e8  # number of timesteps
    n_envs = 32  # number of environment copies simulated in parallel
    n_sample_steps = 128  # number of steps of the environment per sample
    n_mini_batches = 16  # number of training minibatches per update 
                                     # For recurrent policies, should be smaller or equal than number of environments run in parallel.
    n_epochs = 4   # number of training epochs per update
    batch_size = n_envs * n_sample_steps
    mini_batch_size = batch_size // n_mini_batches
    n_updates = math.ceil(total_steps / batch_size)
    assert (batch_size % n_mini_batches == 0)

    curriculum_cursor = mp.Value('i', 16)  # current curriculum difficulty
    curriculum_rewards = deque(maxlen=50)
    curriculum_span = 20  # curriculum span
    curriculum_criterion = 9.5  # curriculum criterion

    save_path = "./data"
    [os.makedirs(f"{save_path}/{dir}") for dir in ["data", "model", "plot", "runs"] if not os.path.exists(f"{save_path}/{dir}")]

    envs = ParallelEnv(n_envs, curriculum_cursor, curriculum_span)
    model  = PPO().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step = 0
    log = pd.DataFrame([], columns=["time", "update", "step", "reward", "average_reward", "curriculum_cursor"])
    writer = SummaryWriter()

    for update in range(1, n_updates+1):

        states, values, actions, log_probabilities, advantages, normalized_advantages, targets, total_reward = envs.sample(model, n_sample_steps, gamma, lamda)

        for _ in range(n_epochs):

            indexes = torch.randperm(batch_size)

            for i in range(0, batch_size, mini_batch_size):
                mini_batch_indexes = indexes[i: i + mini_batch_size]

                loss = compute_loss(
                    model, 
                    states[mini_batch_indexes], values[mini_batch_indexes], 
                    actions[mini_batch_indexes], log_probabilities[mini_batch_indexes], 
                    advantages[mini_batch_indexes], normalized_advantages[mini_batch_indexes], targets[mini_batch_indexes],
                    clip_range, value_coefficient, entropy_coefficient
                )

                # Update weights
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        # Log training information
        step += batch_size
        reward = total_reward/n_envs
        curriculum_rewards.append(reward)
        tail_rewards = log["reward"].tail(99)
        average_reward = (tail_rewards.sum() + reward) / (tail_rewards.count() +1)
        cursor = curriculum_cursor.value
        log.at[update] = [datetime.now(), update, step, reward, average_reward, cursor]
        
        writer.add_scalar('Reward', reward, update)
        writer.add_scalar('Average reward', average_reward, update)
        writer.add_scalar('Curriculum cursor', cursor, update)

        print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {update},{step},({cursor},{cursor+curriculum_span}): {reward:.2f}")

        # Update curriculum cursor
        if update % 10 == 0 and len(curriculum_rewards) >= 30:
            curriculum_average_reward = sum(curriculum_rewards) / len(curriculum_rewards)
            if curriculum_average_reward >= curriculum_criterion:
                curriculum_cursor.value = min(cursor+curriculum_span, 250-curriculum_span)
                curriculum_rewards.clear()

        # Save data
        if update % 122 == 0:
            fig = log["average_reward"].plot().get_figure()
            fig.savefig(f"{save_path}/plot/{step}.png")
            copy_tree("./runs", f"{save_path}/runs")

            torch.save(model.state_dict(), f"{save_path}/model/{step}.pkl")
            log.to_csv(f"{save_path}/data/{step}.csv", index=False, header=True)

    # Save data
    fig = log["average_reward"].plot().get_figure()
    fig.savefig(f"{save_path}/plot/{step}.png")
    copy_tree("./runs", f"{save_path}/runs")
    torch.save(model.state_dict(), f"{save_path}/model/{step}.pkl")
    log.to_csv(f"{save_path}/data/{step}.csv", index=False, header=True)

    envs.close()

if __name__ == '__main__':
    train()

