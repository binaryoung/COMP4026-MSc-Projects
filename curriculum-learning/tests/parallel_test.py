import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.distributions import Categorical

from boxoban_environment import BoxobanEnvironment
from ppo import PPO


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_room(select_map):
    room_fixed = []
    room_state = []

    targets = []
    boxes = []
    for row in select_map:
        room_f = []
        room_s = []

        for e in row:
            if e == '#':
                room_f.append(0)
                room_s.append(0)

            elif e == '@':
                room_f.append(1)
                room_s.append(5)

            elif e == '$':
                boxes.append((len(room_fixed), len(room_f)))
                room_f.append(1)
                room_s.append(3)

            elif e == '.':
                targets.append((len(room_fixed), len(room_f)))
                room_f.append(2)
                room_s.append(2)

            else:
                room_f.append(1)
                room_s.append(1)

        room_fixed.append(room_f)
        room_state.append(room_s)

    return np.array(room_fixed), np.array(room_state)

def build_levels():
    levels = []

    with open(f"test_levels.txt") as f:
        content = f.read()
        content = content[1:]
        for level in content.split(";"):
            level = level.strip().split("\n", 1)
            id, room = level[0].strip(), level[1]
            topology, room = generate_room(room.split("\n"))
            levels.append([id, room, topology])

    return levels

def worker(master):
    level = None
    while True:
        cmd, data = master.recv()
        if cmd == 'step':
            observation, reward, done, info = env.step(data)
            if done:
                (id, room, topology) = level
                env = BoxobanEnvironment(room.copy(), topology.copy())
                observation = env.observation
            master.send((observation, reward, done, info))
        elif cmd == 'reset':
            level = data
            (id, room, topology) = level
            env = BoxobanEnvironment(room.copy(), topology.copy())
            master.send(env.observation)
        elif cmd == 'close':
            master.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.workers = []

        self.master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(n_workers)])

        for worker_end in worker_ends:
            p = mp.Process(target=worker, args=(worker_end,))
            p.daemon = True
            p.start()
            self.workers.append(p)

    def reset(self, level):
        for master_end in self.master_ends:
            master_end.send(('reset', level))
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

    def solve(self, level, model, hidden=None):
        observation = torch.tensor(self.reset(level), device=device)

        for t in range(128):
            with torch.no_grad():
                if hidden is None:
                    pi, _ = model(observation)
                else:
                    pi, _ , hidden= model(observation, hidden)

                p = Categorical(pi)
                action = p.sample()

            observation, reward, done, info = self.step(action.tolist())
            observation = torch.tensor(observation, device=device)
            dones = [x["finished"] for x in info]

            if any(dones) == True:
                return True


        return False


def performOneExperiment(model, hidden=None):
    env = ParallelEnv(10)

    levels = build_levels()
    success = []

    for level in levels:
        id, room, topology = level
        if env.solve(level, model, hidden):
            success.append(id)

        if int(id) % 100 == 0:
            print(f"Tested level {id}")

    return success

def performNExperiments(n, model, hidden=None):
    results = []

    for i in range(n):
        print(f"Perform experiment {i+1}")

        if hidden is None:
            experiment_hidden = None
        else:
            experiment_hidden = (hidden[0].copy(), hidden[1].copy())

        result = performOneExperiment(model, experiment_hidden)
        results.append(result)
    
    count = [len(result) for result in results]
    DF = pd.DataFrame({
        "count": count,
        "solved": results,
    })

    return DF

if __name__ == "__main__":
    model = PPO().to(device)
    model.load_state_dict(torch.load("ppo.pkl"))
    model.eval()

    df = performNExperiments(10, model)
    df.to_csv("result.csv", index=False)
    print(df)
