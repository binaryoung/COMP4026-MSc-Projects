import sys
sys.path.append("./tests/gym-sokoban")
sys.path.append("./src")
import numpy as np
import gym
import gym_sokoban
import pytest
import boxoban_level_collection as collection
from boxoban_environment import BoxobanEnvironment

def test_environment():
    gym_env = gym.make('Boxoban-Train-v0')
    gym_env.set_maxsteps(500)

    for i in range(5000):
        (id, score, trajectory, room, topology) = collection.find(i)

        gym_observation = gym_env.reset(room.copy(), topology.copy())
        gym_observation = gym_env.room_state
        env = BoxobanEnvironment(room.copy(), topology.copy())
        observation = env.room

        # print(gym_observation.astype("uint8"), convert_state(observation))
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
