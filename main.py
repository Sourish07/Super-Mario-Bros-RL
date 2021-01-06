import time

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

import atari_wrappers
from agent import Agent

from sample_wrappers import wrapper

# Setting up environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

env = atari_wrappers.MaxAndSkipEnv(env)
env = atari_wrappers.wrap_deepmind(env, False, True, True, False)

# env = wrapper(env)

state_dimensions = (84, 84, 4)  # Representing four stacked frames, each of size 84 x 84
num_of_actions = env.action_space.n

agent = Agent(state_dimensions, num_of_actions)

# Hyperparameters
num_of_episodes = 10000

rewards = []

load = False
if load:
    num_of_episodes = 1
    agent.load_model()
    agent.epsilon = 0.5

for e in range(num_of_episodes):
    start_time = time.time()

    state = env.reset()

    total_reward = 0
    iter = 0

    done = False
    while not done:
        env.render()
        # if e % 1000 == 0:
        #     env.render()
        start_time = time.time()
        action = agent.choose_action(state)
        print(time.time() - start_time)

        start_time = time.time()
        next_state, reward, done, info = env.step(action=action)
        print(f'env step time: {time.time() - start_time}')

        start_time = time.time()
        agent.memory.append((state, next_state, action, reward, done))
        print(time.time() - start_time)

        start_time = time.time()
        agent.learn()
        print(f'agent learn time {time.time() - start_time}')

        start_time = time.time()
        total_reward += reward
        state = next_state
        iter += 1
        print(time.time() - start_time)

    if e % 100 == 0 and e > 0:
        agent.save_model(e)

    rewards.append(total_reward)

    end_time = time.time()
    print(f'----------Episode #{e+1}, Reward: {total_reward}, Time taken: {end_time - start_time}----------')

