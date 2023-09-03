import torch

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import make_env

import matplotlib.pyplot as plt

import os
from timer import Timer

try:
    os.mkdir("models")
except FileExistsError:
    # delete all files in the models directory
    for file in os.listdir("models"):
        os.remove(os.path.join("models", file))

# delete log.txt
try:
    os.remove("log.txt")
except FileNotFoundError:
    pass

# import logging
# logging.getLogger().setLevel(logging.CRITICAL)

assert torch.cuda.is_available(), "CUDA is not available"
print(torch.cuda.get_device_name(0))

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = False
NUM_OF_EPISODES = 10000


env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

env = make_env(env)

# print("env.observation_space.shape", env.observation_space.shape)
# print("env.action_space.n", env.action_space.n)

agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n)

if not SHOULD_TRAIN:
    agent.load_model("models/model_1375000_iter.pt")
    agent.epsilon = 0.1
    agent.eps_min = 0.0
    agent.eps_decay = 0.0
# print("Action space:", env.action_space, type(env.action_space)) # Action space: Discrete(5) <class 'gym.spaces.discrete.Discrete'>

# print("Observation space:", env.observation_space, type(env.observation_space)) # Observation space: Box(0, 255, (240, 256, 3), uint8) <class 'gym.spaces.box.Box'>
# Minimum value: 0 and Maximum value: 255, 3 channels mean RGB, height: 240, width: 256

env.reset() # Order of reset being called is reversed (from the order in which the wrappers are applied)
next_state, reward, done, trunc, info = env.step(action=0)
# next_state is of type numpy.ndarray, shape: (240, 256, 3),
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

rewards = []
timer = Timer()
timer.start()

for i in range(NUM_OF_EPISODES):
    with open("log.txt", "a") as f:
        f.write("Episode: " + str(i) + "\n")
    
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info  = env.step(a)
        
        if done:
            with open("log.txt", "a") as f:
                f.write(f"Info: {info} \n")

        total_reward += reward

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

    with open("log.txt", "a") as f:
        f.write("Total reward: " + str(total_reward) + "\n")
        f.write("Epsilon: " + str(agent.epsilon) + "\n")
        f.write("Size of replay buffer: " + str(len(agent.replay_buffer)) + "\n")
        f.write("Learn step counter: " + str(agent.learn_step_counter) + "\n")
        f.write("\n\n")

    if SHOULD_TRAIN and (i + 1) % 1000 == 0:
        agent.save_model("models/model_" + str(i + 1) + "_iter.pt")

    print("Total reward:", total_reward)
    rewards.append(total_reward)

env.close()
print("Total time taken:", timer.get())

average_reward = sum(rewards) / len(rewards)
print("Average reward:", average_reward)

plt.plot(rewards)
plt.savefig("rewards.png")
# plt.show()
