import sys
sys.path.append("./")
import time

import torch
from torch.distributions import Categorical
import numpy as np
from boxoban_env import BoxobanEnv

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


def convert_observation(observation, topology):
    room = observation.copy()
    room[observation == 3] = 4
    room[observation == 4] = 3
    room[(room == 5) & (topology == 2)] = 6

    return to_observation(room)

def to_observation(room):
    wall = np.zeros((10, 10), dtype=np.float32)
    empty = np.zeros((10, 10), dtype=np.float32)
    target = np.zeros((10, 10), dtype=np.float32)
    box = np.zeros((10, 10), dtype=np.float32)
    box_on_target = np.zeros((10, 10), dtype=np.float32)
    player = np.zeros((10, 10), dtype=np.float32)
    player_on_target = np.zeros((10, 10), dtype=np.float32)

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

    return np.stack((
        wall,
        empty,
        target,
        box,
        box_on_target,
        player,
        player_on_target,
    ), axis=0)

def build_levels():
    levels = []

    with open(f"tests/test_levels.txt") as f:
        content = f.read()
        content = content[1:]
        for level in content.split(";"):
            level = level.strip().split("\n", 1)
            id, room = level[0].strip(), level[1]
            topology, room = generate_room(room.split("\n"))
            levels.append([id, room, topology])

    return levels

def render_level(model, level):
    id, room, topology = level

    env = BoxobanEnv()
    env.set_maxsteps(128)

    observation = env.reset(room.copy(), topology.copy())
    observation = env.room_state

    while True:
        env.render(mode='human')

        with torch.no_grad():
            observation = convert_observation(observation, topology)
            pi, _ = model(torch.tensor(observation, device=device).unsqueeze(0))
            p = Categorical(pi)
            observation, reward, done, info = env.step(p.sample().item()+1)
            observation = env.room_state
        
        time.sleep(0.3)

        if done:
            env.render()
            break

    env.close()

def solve_level(model, level):
    id, room, topology = level

    for i in range(10):
        env = BoxobanEnvironment(room.copy(), topology.copy())
        observation = env.observation

        while True:
            with torch.no_grad():
                pi, _ = model(torch.tensor(observation, device=device).unsqueeze(0))
                p = Categorical(pi)
                observation, reward, done, info = env.step(p.sample())

            if done:
                if info["finished"]:
                    return True

                break

    return False
    
def test_ppo():
    levels = build_levels()
    success = []

    model = PPO().to(device)
    model.load_state_dict(torch.load("models/ppo.pkl"))
    model.eval()

    for level in levels:
        id, room, topology = level
        if solve_level(model, level):
            success.append(id)
        print(f"Test level {id}")


    return success

def render_ppo(id):
    level = build_levels()[id]
    
    model = PPO().to(device)
    model.load_state_dict(torch.load("models/ppo.pkl"))
    model.eval()

    render_level(model, level)

if __name__ == "__main__":
    render_ppo(0)

    success = test_ppo()
    print(len(success), success)
