import sys
sys.path.append("./gym-sokoban-master")
import time
import gym_sokoban
import gym

# Before you can make a Sokoban Environment you need to call:
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package
env_name = 'Sokoban-v1'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))

for i_episode in range(1):  # 20
    observation, score, trajectory = env.reset()
    print(score,trajectory, len(trajectory))

    i = 1
    for action in trajectory:  # 100
        env.render(mode='human')

        # Sleep makes the actions visible for users
        time.sleep(99999)
        observation, reward, done, info = env.step(action)
        print(info)

        print(ACTION_LOOKUP[action], reward, done, info)
        i += 1
        if done:
            print("Episode finished after {} timesteps".format(i))
            env.render()
            break

    env.close()

time.sleep(10)
