import sys
sys.path.append("./src")
import numpy as np
from boxoban_env import BoxobanEnv
import boxoban_level_collection as collection
from boxoban_environment import BoxobanEnvironment

def test_environment():
    gym_env = BoxobanEnv()
    gym_env.set_maxsteps(500)

    for i in range(5000):
        (id, score, trajectory, room, topology) = collection.find(i)

        gym_observation = gym_env.reset(room.copy(), topology.copy())
        gym_observation = gym_env.room_state
        env = BoxobanEnvironment(room.copy(), topology.copy())
        env.reward_per_step = -0.1
        env.max_steps = 500
        observation = env.room

        assert np.array_equal(gym_observation.astype("uint8"), convert_state(observation)) == True

        for action in trajectory:
            gym_observation, gym_reward, gym_done, gym_info = gym_env.step(action+1)
            gym_observation = gym_env.room_state
            observation, reward, done, info = env.step(action)
            observation = env.room

            assert np.array_equal(gym_observation.astype("uint8"), convert_state(observation)) == True
            assert gym_reward == reward
            assert gym_done== done

            if gym_done:
                assert gym_done == done
                assert gym_info.get("all_boxes_on_target", False) == True
                assert info["finished"] == True
                break

        print(f'{id}: pass')

def convert_state(state):
    room = state.copy()
    room[state == 3] = 4
    room[state == 4] = 3
    room[state == 6] = 5

    return room

def run_gym_levels(number):
    env = BoxobanEnv()
    env.set_maxsteps(500)

    for i in range(number):
        (id, score, trajectory, room, topology) = collection.find(i)
        observation = env.reset(room, topology)

        for action in trajectory:
            observation, reward, done, info = env.step(action+1)

            if done:
                break

        assert info.get("all_boxes_on_target", False) == True

def run_env_levels(number):
    for i in range(number):
        (id, score, trajectory, room, topology) = collection.find(i)
        env = BoxobanEnvironment(room, topology)
        env.max_steps = 500

        for action in trajectory:
            observation, reward, done, info = env.step(action)

            if done:
                break
        
        assert info["finished"] == True

def test_gym_performace(benchmark):
    benchmark(run_gym_levels, 5)


def test_env_performace(benchmark):
    benchmark(run_env_levels, 5)

