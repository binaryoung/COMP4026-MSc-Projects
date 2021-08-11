import torch
import torch.multiprocessing as mp

import boxoban_level_collection as collection
from boxoban_environment import BoxobanEnvironment

def build(i, start, end):
    states = []
    actions = []

    for id in range(start, end):
        level_states = []
        level_actions = []

        (_, _, trajectory, room, topology) = collection.find(id)

        env = BoxobanEnvironment(room, topology)

        for action in trajectory:
            level_states.append(torch.tensor(env.room.copy()))
            level_actions.append(torch.tensor(action))
            observation, reward, done, info = env.step(action)

        assert done == True and info["finished"] == True
        assert len(level_states) == len(trajectory) and len(level_actions) == len(trajectory)

        level_states = torch.stack(level_states, dim=0)
        level_actions = torch.stack(level_actions, dim=0)
        states.append(level_states)
        actions.append(level_actions)

        if id % 1000 == 0:
            print(f"Builded level {id//1000}")
    
    assert len(states) == len(actions) and len(states) == 1000000//10

    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    assert states.size(0) == actions.size(0)

    torch.save(states, f"samples/states_{i}.pt")
    torch.save(actions, f"samples/actions_{i}.pt")


if __name__ == "__main__":
    workers = []
    start_id = range(0, 1000001, 1000000//10)

    for i in range(len(start_id)-1):
        p = mp.Process(target=build, args=(i, start_id[i], start_id[i+1]))
        p.start()
        workers.append(p)

    [worker.join() for worker in workers]
