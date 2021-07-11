import sys
sys.path.append("./tests/gym_sokoban")
import hashlib
from os import listdir
import gym
import gym_sokoban
import pandas as pd
import matplotlib.pyplot as plt
import time


def build_levels():
    levels = []

    files = [f for f in listdir("levels")]
    for i, file in enumerate(files):
        with open(f"levels/{file}") as f:
            content = f.read()
            content = content[1:]
            for level in content.split(";"):
                level = level.strip().split("\n", 1)
                meta, room = level[0].split(" "), level[1]
                id, score, trajectory = int(meta[0])+i*1000, int(meta[1]), list(map(int, meta[2].split(",")))
                hash = hashlib.sha1(room.encode('utf-8')).hexdigest()
                levels.append([id, score, trajectory, len(trajectory), hash, room])
    
    return levels

def test_levels():
    env = gym.make('Boxoban-Train-v0')
    env.set_maxsteps(1000)

    levels = build_levels()

    for level in levels:
        id, room, trajectory = level[0], level[5], level[2]
        observation = env.reset(room)

        for action in trajectory:
            # env.render(mode='human')
            # time.sleep(0.3)

            observation, reward, done, info = env.step(action+1)

            if done:
                # env.render()
                break
        
        print(f'{id}: {info.get("all_boxes_on_target", False)}')
        assert info.get("all_boxes_on_target", False) == True
        # env.close()

if __name__ == "__main__":
    levels = build_levels()
    df = pd.DataFrame(levels, columns=['id', 'score', 'trajectory', 'steps', 'hash', 'room'])
    print(df["score"].describe())
    print(df["steps"].describe())
    df["score"].plot.hist(bins=100)
    plt.show()
