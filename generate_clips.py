"""
This script was used to generate the visuals for the Deep Q Learning Algorithm's progress. To simply train and see
model, use main.py
"""
import os
import shutil
import time
import subprocess

import cv2
import numpy as np
import gym_super_mario_bros
import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from pytorch_agent import Agent

from wrappers import make_env

# Setting up environment
env_name = 'SuperMarioBros-1-1-v0'
env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, RIGHT_ONLY)
#
# env = atari_wrappers.MaxAndSkipEnv(env)
# env = atari_wrappers.wrap_deepmind(env, False, True, False, False)

env = make_env(env)

agent = Agent(env.observation_space.shape,
              env.action_space.n,
              env_name=env_name)
#agent.load_models(10000)
#agent.epsilon = 0.01
start = time.time()

# Hyperparameters
num_of_episodes = 20000

scores, steps_array = [], []

load = True
if load:
    num_of_episodes = 10
    agent.load_models(20000)
    agent.epsilon = 0.1

total_steps = 0
root_directory = os.getcwd()

for e in range(0, num_of_episodes):

    start_time = time.time()

    state = env.reset()

    score = 0
    num_of_iterations = 0

    done = False

    os.mkdir(f'game{e + 1}')

    while not done:
        action = agent.choose_action(state)

        if action == 0:
            controller = cv2.imread("controller states/noop.png")
        elif action == 1:
            controller = cv2.imread("controller states/right.png")
        elif action == 2:
            controller = cv2.imread("controller states/right a.png")
        elif action == 3:
            controller = cv2.imread("controller states/right b.png")
        else:
            controller = cv2.imread("controller states/right a b.png")

        if load:
            frame = env.render(mode='rgb_array')
            frame = np.flip(frame, axis=-1)

            # resizing frame
            frame_scale_factor = 3
            width = int(frame.shape[1] * frame_scale_factor)
            height = int(frame.shape[0] * frame_scale_factor)

            frame_dim = (width, height)
            frame = cv2.resize(frame, frame_dim, interpolation=cv2.INTER_AREA)

            # resizing controller
            controller_scale_factor = 0.25
            width = int(controller.shape[1] * controller_scale_factor)
            height = int(controller.shape[0] * controller_scale_factor)
            #print("ERROR: ACTION IS: ", action)
            controller_dim = (width, height)
            controller = cv2.resize(controller, controller_dim, interpolation=cv2.INTER_AREA)

            controller_height, controller_width, _ = controller.shape
            result = cv2.addWeighted(frame[-controller_height:, -controller_width:], 0, controller, 1, 0)
            frame[-controller_height:, -controller_width:] = result

            cv2.imwrite(f'game{e + 1}/frame{num_of_iterations}.jpg', frame)

        next_state, reward, done, info = env.step(action)
        if done:
            #if info['flag_get']:
            print(f"Done!")
            os.chdir(root_directory + f'\\game{e + 1}')
            subprocess.call(['ffmpeg', '-framerate', '24', '-i', 'frame%d.jpg', f'../game{e + 1}.mp4'])
            os.chdir(root_directory)

        if not load:
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

        score += reward
        state = next_state
        num_of_iterations += 1
        total_steps += 1


    if (e + 1) % 1000 == 0 and not load:
        agent.save_models(e + 1)

    scores.append(score)
    steps_array.append(total_steps)

    mean_reward = np.mean(scores[-100:])

    end_time = time.time()
    print(f'Episode #{e + 1}, '
          f'Score: {score}, '
          f'Number of Iterations: {num_of_iterations}, '
          f'Mean Reward: {mean_reward}, '
          f'Epsilon: {np.round(agent.epsilon, 4)}, '
          f'Time taken: {end_time - start_time}')

plt.plot(steps_array, scores)
plt.show()

for e in range(num_of_episodes):
    shutil.rmtree(root_directory + f'\\game{e + 1}')