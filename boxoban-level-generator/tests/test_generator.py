import hashlib
from os import listdir
from boxoban_env import BoxobanEnv
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

def build_test_levels():
    levels = []

    with open(f"tests/test_levels.txt") as f:
        content = f.read()
        content = content[1:]
        for level in content.split(";"):
            level = level.strip().split("\n", 1)
            id, room = level[0].strip(), level[1]
            hash = hashlib.sha1(room.encode('utf-8')).hexdigest()
            levels.append([id, room, hash])

    return levels

def test_levels():
    env = BoxobanEnv()
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

def analysis_levels():
    levels = build_levels()
    df = pd.DataFrame(levels, columns=['id', 'score', 'trajectory', 'steps', 'hash', 'room'])
    print(df["score"].describe())
    print(df["steps"].describe())
    print(df[df.duplicated(["hash"], keep="first")].describe())
    score = df["score"]
    score[score <= 250].plot.hist(bins=20)
    # df["steps"].plot.hist(bins=100)
    print(df[df["steps"] > 120].describe())
    plt.show()

def analysis_duplicated_levels():
    levels = build_levels()
    df = pd.DataFrame(levels, columns=['id', 'score', 'trajectory', 'steps', 'hash', 'room'])

    test_levels = build_test_levels()
    duplicated = [(df.hash == hash).sum() for (_, _, hash) in test_levels]

    print(sum(duplicated))

if __name__ == "__main__":
    analysis_levels()
    analysis_duplicated_levels()
