import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
import boxoban_level_collection as collection
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
            nn.Linear(2304, 256),  # 64, 6, 6
            nn.ReLU()
        )

        self.actor_head = nn.Linear(256, 7)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        actor = F.softmax(self.actor_head(x), dim=-1)
        critic = self.critic_head(x)

        return actor, critic
            

def collection_worker(queue):
    while True:
        queue.put(collection.random())

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
        # self.collection.join()

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

        for t in reversed(range(steps)):
            mask = 1.0 - dones[:, t].int()

            delta = rewards[:, t] + gamma * last_value * mask - values[:, t]

            last_advantage = delta + gamma * lamda * last_advantage * mask

            advantages[:, t] = last_advantage

            last_value = values[:, t]


        states = states.reshape(-1, 7, 10, 10)
        values = values.reshape(-1)
        actions = actions.reshape(-1)
        log_probabilities = log_probabilities.reshape(-1)
        advantages = advantages.reshape(-1)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        targets = advantages + values

        return states, values, actions, log_probabilities, advantages, normalized_advantages, targets

def train():
    learning_rate = 1e-4
    gamma = 0.99
    lamda = 0.95
    clip_range = 0.2
    value_coefficient = 0.5
    entropy_coefficient = 0.01
    max_grad_norm = 0.5

    total_steps = 1e8  # number of timesteps
    n_envs = 2  # number of environment copies simulated in parallel
    n_sample_steps = 128  # number of steps of the environment per sample
    n_mini_batches = 4  # number of training minibatches per update 
                                     # For recurrent policies, should be smaller or equal than number of environments run in parallel.
    n_epochs = 4   # number of training epochs per update
    batch_size = n_envs * n_sample_steps
    mini_batch_size = batch_size // n_mini_batches
    assert (batch_size % n_mini_batches == 0)
    n_updates = math.ceil(total_steps / batch_size)


    envs = ParallelEnv(n_envs)
    model  = PPO().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0

    for update in range(1, n_updates+1):

        states, values, actions, log_probabilities, advantages, normalized_advantages, targets = envs.sample(
            model, n_sample_steps, gamma, lamda)

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

                # for pg in optimizer.param_groups:
                #     pg['lr'] = learning_rate

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
        
        steps += batch_size
        print(steps)


def compute_loss(model, states, values, actions, log_probabilities, advantages, normalized_advantages, targets, clip_range,  value_coefficient, entropy_coefficient):
    pi, v = model(states)
    v = v.squeeze(1)

    p = Categorical(pi)
    log_actions = p.log_prob(actions)

    ratio = torch.exp(log_actions - log_probabilities)
    surrogate1 = ratio * normalized_advantages
    surrogate2 = ratio.clamp(1-clip_range, 1+clip_range) * normalized_advantages
    policy_loss = torch.min(surrogate1, surrogate2).mean()

    clipped_value = values + (v - values).clamp(-clip_range, clip_range)
    value_loss1 = (v - targets).square()
    value_loss2 = (clipped_value - targets).square()
    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

    entropy_loss = p.entropy().mean()

    loss = -(policy_loss - value_coefficient * value_loss + entropy_coefficient * entropy_loss)

    return loss


if __name__ == '__main__':
    # envs = ParallelEnv(2)
    # model = PPO().to(device)
    # for i in range(10):
    #     states, values, actions, log_probabilities, advantages = envs.sample( model, 128)
    #     print(states.shape, values.shape, actions.shape, log_probabilities.shape, advantages.shape)
    #     print(values[:5], actions[:5])

    train()

