import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import make_env

import logging
logging.getLogger().setLevel(logging.CRITICAL)

assert torch.cuda.is_available(), "CUDA is not available"
print(torch.cuda.get_device_name(0))

ENV_NAME = 'SuperMarioBros-1-1-v0'
DISPLAY = False
NUM_OF_EPISODES = 1

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

env = make_env(env)

print("env.observation_space.shape", env.observation_space.shape)
print("env.action_space.n", env.action_space.n)

agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, epsilon=0)

# print("Action space:", env.action_space, type(env.action_space)) # Action space: Discrete(5) <class 'gym.spaces.discrete.Discrete'>

# print("Observation space:", env.observation_space, type(env.observation_space)) # Observation space: Box(0, 255, (240, 256, 3), uint8) <class 'gym.spaces.box.Box'>
# Minimum value: 0 and Maximum value: 255, 3 channels mean RGB, height: 240, width: 256

env.reset() # Order of reset being called is reversed (from the order in which the wrappers are applied)
next_state, reward, done, trunc, info = env.step(action=0)
# next_state is of type numpy.ndarray, shape: (240, 256, 3),
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

for i in range(NUM_OF_EPISODES):
    state, _ = env.reset()
    a = agent.choose_action(state)
    print("a", a)