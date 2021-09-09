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
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import boxoban_level_collection as levelCollection
from boxoban_environment import BoxobanEnvironment


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Model
class GAIL(nn.Module):
    def __init__(self):
        super(GAIL, self).__init__()

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

        self.discriminator_encoder = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2304, 256),  # 64, 6, 6
            nn.ReLU()
        )

        self.discriminator = nn.Sequential(
            nn.Linear(264, 128),  # 256 + 8
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        actor = F.softmax(self.actor_head(x), dim=-1)
        critic = self.critic_head(x)

        return actor, critic
    
    """
    Expert -> 0
    Policy -> 1
    """
    def discriminate(self, states, actions):
        states = self.discriminator_encoder(states)
        actions = F.one_hot(actions, num_classes=8).float()

        x = torch.cat((states, actions), dim=1)

        x = self.discriminator(x)

        return x

    # Compute discriminator reward
    def discriminator_reward(self, states, actions):
        with torch.no_grad():
            logits = self.discriminate(states, actions)

            return -F.logsigmoid(logits)


# Expert demonstration dataset
class ExpertDataset(Dataset):
    def __init__(self):
        self.states = torch.load("samples/states.pt", map_location=device)
        self.actions = torch.load("samples/actions.pt", map_location=device)

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]

        state = self.to_observation(state)

        return state, action

    def __len__(self):
        return self.actions.size(0)

    @staticmethod
    def to_observation(room):
        wall = torch.zeros((10, 10), dtype=torch.float32, device=device)
        empty = torch.zeros((10, 10), dtype=torch.float32, device=device)
        target = torch.zeros((10, 10), dtype=torch.float32, device=device)
        box = torch.zeros((10, 10), dtype=torch.float32, device=device)
        box_on_target = torch.zeros((10, 10), dtype=torch.float32, device=device)
        player = torch.zeros((10, 10), dtype=torch.float32, device=device)
        player_on_target = torch.zeros((10, 10), dtype=torch.float32, device=device)

        for i, row in enumerate(room):
            for j, state in enumerate(row):
                if state == 0:
                    wall[i, j] = 1
                elif state == 1:
                    empty[i, j] = 1
                elif state == 2:
                    target[i, j] = 1
                elif state == 3:
                    box[i, j] = 1
                elif state == 4:
                    box_on_target[i, j] = 1
                elif state == 5:
                    player[i, j] = 1
                elif state == 6:
                    player_on_target[i, j] = 1

        return torch.stack((
            wall,
            empty,
            target,
            box,
            box_on_target,
            player,
            player_on_target,
        ), dim=0)

# Level collection process
def collection_worker(queue):
    while True:
        queue.put(levelCollection.random())

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
        discriminator_rewards = torch.zeros((self.n_workers, steps), dtype=torch.float32, device=device)

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

                discriminator_rewards[:, t] = model.discriminator_reward(observation, action).squeeze(-1)

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

            delta = (1 * rewards[:, t] + 0.01 * discriminator_rewards[:, t]) + gamma * last_value * mask - values[:, t]

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
        discriminator_total_reward = discriminator_rewards.sum().item()

        return states, values, actions, log_probabilities, advantages, normalized_advantages, targets, total_reward, discriminator_total_reward


# Compute discriminator loss
def compute_discriminator_loss(model, expert_states, expert_actions, states, actions):
    expert_predictions = model.discriminate(expert_states, expert_actions)
    policy_predictions = model.discriminate(states, actions.long())

    expert_targets = torch.zeros_like(expert_predictions, device=device)
    policy_targets = torch.ones_like(policy_predictions, device=device)

    expert_loss = F.binary_cross_entropy_with_logits(expert_predictions, expert_targets)
    policy_loss = F.binary_cross_entropy_with_logits(policy_predictions, policy_targets)

    loss = expert_loss + policy_loss

    return loss

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
    learning_rate = 3e-4  # learning rate
    gamma = 0.99 # gamma
    lamda = 0.95  # GAE lambda
    clip_range = 0.1 # surrogate objective clip range
    value_coefficient = 0.5 # value coefficient in loss function
    entropy_coefficient = 0.01 # entropy coefficient in loss function
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

    n_discriminator_epochs = 5   # number of discriminator training epochs per update
    discriminator_batch_size = 64  # number of discriminator training batch size per update

    save_path = "./data"
    [os.makedirs(f"{save_path}/{dir}") for dir in ["data", "model", "plot", "runs", "discriminator", "loss"] if not os.path.exists(f"{save_path}/{dir}")]

    envs = ParallelEnv(n_envs)
    model  = GAIL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    expert_dataset = ExpertDataset()
    expert_dataloader = DataLoader(dataset=expert_dataset, batch_size=discriminator_batch_size, shuffle=True)
    expert_dataloader = iter(expert_dataloader)

    step = 0
    log = pd.DataFrame([], columns=[
        "time", "update", "step", 
        "reward", "average_reward", 
        "discriminator_reward", "discriminator_average_reward",
        "discriminator_loss", "discriminator_average_loss"
    ])
    writer = SummaryWriter()

    for update in range(1, n_updates+1):

        states, values, actions, log_probabilities, advantages, normalized_advantages, targets, total_reward, discriminator_total_reward = envs.sample(model, n_sample_steps, gamma, lamda)

        discriminator_loss = 0
        discriminator_indexes = torch.randperm(batch_size)

        # Train the discriminator
        for i in range(0, discriminator_batch_size * n_discriminator_epochs, discriminator_batch_size):
            discriminator_batch_indexes = discriminator_indexes[i: i + discriminator_batch_size]

            expert_states, expert_actions = next(expert_dataloader)

            loss = compute_discriminator_loss(
                model,
                expert_states, expert_actions,
                states[discriminator_batch_indexes], actions[discriminator_batch_indexes]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            discriminator_loss += loss.item()

        # Train the model
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

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        # Log training information
        step += batch_size
        reward = total_reward/n_envs
        tail_rewards = log["reward"].tail(99)
        average_reward = (tail_rewards.sum() + reward) / (tail_rewards.count() +1)
        discriminator_reward = discriminator_total_reward/n_envs
        discriminator_tail_rewards = log["discriminator_reward"].tail(99)
        discriminator_average_reward = (discriminator_tail_rewards.sum() + discriminator_reward) / (discriminator_tail_rewards.count() +1)
        discriminator_loss = discriminator_loss/n_discriminator_epochs
        discriminator_tail_loss = log["discriminator_loss"].tail(99)
        discriminator_average_loss = (discriminator_tail_loss.sum() + discriminator_loss) / (discriminator_tail_loss.count() +1)
        log.at[update] = [
            datetime.now(), update, step, 
            reward, average_reward, 
            discriminator_reward, discriminator_average_reward,
            discriminator_loss, discriminator_average_loss
        ]
        
        writer.add_scalar('Reward', reward, update)
        writer.add_scalar('Average reward', average_reward, update)
        writer.add_scalar('Discriminator reward', discriminator_reward, update)
        writer.add_scalar('Discriminator average reward', discriminator_average_reward, update)
        writer.add_scalar('Discriminator loss', discriminator_loss, update)
        writer.add_scalar('Discriminator average loss', discriminator_average_loss, update)

        print(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {update},{step}: {reward:.2f}, {discriminator_reward:.3f}, {discriminator_loss:.3f}")

        # Save data
        if update % 122 == 0:
            fig = log["average_reward"].plot().get_figure()
            fig.savefig(f"{save_path}/plot/{step}.png")
            plt.clf()
            fig = log["discriminator_average_reward"].plot().get_figure()
            fig.savefig(f"{save_path}/discriminator/{step}.png")
            plt.clf()
            fig = log["discriminator_average_loss"].plot().get_figure()
            fig.savefig(f"{save_path}/loss/{step}.png")
            plt.clf()
            copy_tree("./runs", f"{save_path}/runs")

            torch.save(model.state_dict(), f"{save_path}/model/{step}.pkl")
            log.to_csv(f"{save_path}/data/{step}.csv", index=False, header=True)

    # Save data
    fig = log["average_reward"].plot().get_figure()
    fig.savefig(f"{save_path}/plot/{step}.png")
    plt.clf()
    fig = log["discriminator_average_reward"].plot().get_figure()
    fig.savefig(f"{save_path}/discriminator/{step}.png")
    plt.clf()
    fig = log["discriminator_average_loss"].plot().get_figure()
    fig.savefig(f"{save_path}/loss/{step}.png")
    plt.clf()
    copy_tree("./runs", f"{save_path}/runs")
    torch.save(model.state_dict(), f"{save_path}/model/{step}.pkl")
    log.to_csv(f"{save_path}/data/{step}.csv", index=False, header=True)

    envs.close()

if __name__ == '__main__':
    train()

