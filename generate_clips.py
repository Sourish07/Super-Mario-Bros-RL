import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from agent import Agent

from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

import os
from PIL import Image

# Modified SkipFrame wrapper to log frames and actions
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
        self.counter = 0
        self.frames_log = []
        self.actions_log = []
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            self.frames_log.append(next_state.copy())
            self.actions_log.append(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info
    
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.frames_log = [state.copy()]
        self.actions_log = [0]
        return state, info

def apply_wrappers(env):
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env


ENV_NAME = 'SuperMarioBros-1-1-v0'
NUM_OF_EPISODES = 50000
controllers = [Image.open(f"controllers/{i}.png") for i in range(5)]

env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# agent.load_model("models/folder_name/model_80000_iter.pt")

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    rewards = 0
    while not done:
        action = agent.choose_action(state)
        frame = env.render()
        new_state, reward, done, truncated, info  = env.step(action)
        rewards += reward

        agent.store_in_memory(state, action, reward, new_state, done)
        agent.learn()

        state = new_state

        if done:
            print(f"Episode: {i}, Reward: {rewards}")
            if info["flag_get"]:
                os.makedirs(f"games/game_{i}", exist_ok=True)
                frame_skip_env = env.env.env.env
                frames_log = frame_skip_env.frames_log
                actions_log = frame_skip_env.actions_log
                for j, (frame, action) in enumerate(zip(frames_log, actions_log)):
                    # upscale frame
                    scaling_factor = 10
                    new_dims = (frame.shape[1] * scaling_factor, frame.shape[0] * scaling_factor)
                    frame = Image.fromarray(frame).resize(new_dims, Image.NEAREST)
                    frame.save(f"games/game_{i}/frame_{j}.png")

                    controllers[action].save(f"games/game_{i}/controller_{j}.png")
        
        if i % 5000 == 0 and i > 0:
            agent.save_model(f"models_mse_loss/model_{i}_iter.pt")

env.close()

