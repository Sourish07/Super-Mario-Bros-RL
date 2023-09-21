import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from wrappers import apply_wrappers
from agent import Agent
import os

import matplotlib.pyplot as plt

ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_OF_EPISODES = 50000

env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

agent.load_model("models with smoothl1loss/good_checkpoints/model_50000_iter.pt")

print()
# exit(0)

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    rewards = 0
    while not done:
        action = agent.choose_action(state)

        frame = env.render()

        new_state, reward, done, truncated, info  = env.step(action)
        
        rewards += reward
        if done or info["time"] < 100:
            print(f"Info: {info}, Episode num: {i}, Reward: {rewards} \n")
            if info["flag_get"] or rewards > 2000:
                os.makedirs(f"games/game_{i}", exist_ok=True)
                for j, frame in enumerate(env.env.env.env.frames):
                    if not info["flag_get"] and j % 4 != 0:
                        continue
                    plt.gca().set_axis_off()
                    plt.imshow(frame)
                    plt.savefig(f"games/game_{i}/frame{j}.png", bbox_inches = 'tight', pad_inches = 0)
                    plt.close()

            break
        
        agent.store_in_memory(state, action, reward, new_state, done)
        agent.learn()

        if i % 10000 == 0:
            agent.save_model(f"models with smoothl1loss/good_checkpoints/model_{50000 + i}_iter.pt")


        state = new_state

env.close()

