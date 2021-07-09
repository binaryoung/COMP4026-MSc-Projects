import hashlib
from os import listdir
import sys
sys.path.append("./gym-sokoban-master")
import time
import gym_sokoban
import gym
import pandas as pd
import matplotlib.pyplot as plt

# Before you can make a Sokoban Environment you need to call:
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package

levels = []

files = [f for f in listdir("levels")]
for i,file in enumerate(files):
    with open(f"levels/{file}") as f:
        content = f.read()
        content = content[1:]
        for level in content.split(";"):
            level = level.strip().split("\n", 1)
            meta,room = level[0].split(" "),level[1]
            id, score, trajectory = int(meta[0])+i*1000, int(meta[1]), list(map(int, meta[2].split(",")))
            hash = hashlib.sha1(room.encode('utf-8')).hexdigest()
            levels.append([id, score, trajectory,len(trajectory), hash, room])

# df = pd.DataFrame(levels, columns=['id', 'score', 'trajectory', 'steps', 'hash', 'room'])
# print(df["score"].describe())
# print(df["steps"].describe())
# df["score"].plot.hist(bins=100)
# plt.show()

env = gym.make('Boxoban-Train-v0')
env.set_maxsteps(1000)
ACTION_LOOKUP = env.unwrapped.get_action_lookup()
id = -1
info = {"all_boxes_on_target": True}
fail= []
success = []

for level in levels:
    print(f'{id}: {info.get("all_boxes_on_target", False)}')
    if info.get("all_boxes_on_target", False) == False:
        fail.append(id)
    if info.get("all_boxes_on_target", False) == True:
        success.append(id)

    id,room, trajectory = level[0],level[5], level[2]
    observation = env.reset(room)

    for action in trajectory: 
        # env.render(mode='human')

        # Sleep makes the actions visible for users
        # time.sleep(0.3)
        observation, reward, done, info = env.step(action+1)

        # print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            # print("Episode finished")
            # env.render()
            break
    
    # env.close()

print(id,info)
print(fail)
print(len(success))
