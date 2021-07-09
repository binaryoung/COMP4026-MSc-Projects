import sys
sys.path.append("./tests/gym-sokoban")
import time
import random
import boxoban_level_collection as collection
import timeit
import pytest
import pandas as pd
import gym
import gym_sokoban

def test_find():
    level = collection.find(0)
    assert len(level) == 5
    # print(level)

    with pytest.raises(Exception) as e:
        collection.find(999999999999999999)
    assert str(e.value) == "Id 999999999999999999 not exist"

def test_range(benchmark):
    level = benchmark(collection.range, 400, 500)
    assert len(level) == 5
    # print(level)

    with pytest.raises(Exception) as e:
        collection.range(0, 1)
    assert str(e.value) == "Range 0-1 is empty"

def test_random(benchmark):
    level = benchmark(collection.random)
    assert len(level) == 5
    # print(level)

# benchmark into_pyarray to_pyarray
# into_pyarray 2.5756914 2.5101231 2.5817606
# to_pyarray 2.4669522 2.4064948 2.387741
# print(timeit.timeit("collection.random()", "import boxoban_level_collection as collection"))

def build_level_df():
    levels = []
    for id in range(5000):
        levels.append(collection.find(id))
        
    df = pd.DataFrame(levels, columns=[
                      'id', 'score', 'trajectory', 'room', 'topology'])
    
    return df,levels

df, levels = build_level_df()

def random_level():
    return random.choice(levels)

def range_level(min, max):
    return df[df['score'].between(min, max, inclusive=True)].sample(1)

def test_random_level(benchmark):
    level = benchmark(random_level)
    assert len(level) == 5

def test_range_level(benchmark):
    level = benchmark(range_level, 400, 500)
    assert len(level) == 1

def benchmark_range():
    # 0.03599905967712402
    # 6.858001470565796
    start = time.time()
    for _ in range(10000):
        collection.range(400, 500)
    end = time.time()
    print(end-start)

    start = time.time()
    for _ in range(10000):
        range_level(400, 500)
    end = time.time()
    print(end-start)

def test_levels():
    env = gym.make('Boxoban-Train-v0')
    env.set_maxsteps(1000)

    id = -1
    info = {"all_boxes_on_target": True}

    for i in range(5000):
        print(f'{id}: {info.get("all_boxes_on_target", False)}')
        assert info.get("all_boxes_on_target", False) == True

        (id, score, trajectory, room, topology) = collection.find(i)
        observation = env.reset(room, topology)

        for action in trajectory:
            observation, reward, done, info = env.step(action+1)

            if done:
                break
    
        # env.close()

    assert info.get("all_boxes_on_target", False) == True
