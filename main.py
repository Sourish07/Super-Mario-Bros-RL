import torch
import gym

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from nes_py.wrappers import JoypadSpace

ENV_NAME = 'SuperMarioBros-1-1-v0'
DISPLAY = False

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

print("Action space:", env.action_space, type(env.action_space)) # Action space: Discrete(5) <class 'gym.spaces.discrete.Discrete'>

print("Observation space:", env.observation_space, type(env.observation_space)) # Observation space: Box(0, 255, (240, 256, 3), uint8) <class 'gym.spaces.box.Box'>
# Minimum value: 0 and Maximum value: 255, 3 channels mean RGB, height: 240, width: 256

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
# next_state is of type numpy.ndarray, shape: (240, 256, 3),
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")