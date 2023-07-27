import gym
import numpy as np
import torch
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


# Trying to find the max color between two consecutive frames due to flickering
class MaxColor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frame_buffer = np.zeros((2, *self.observation_space.shape), dtype=self.observation_space.dtype)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(2):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            self.frame_buffer[i] = next_state
            if done:
                break
        max_frame = self.frame_buffer.max(axis=0)
        return max_frame, total_reward, done, trunc, info
    
    # Need to reset the frame_buffer otherwise a frame from the previous episode will be used
    def reset(self, **kwargs):
        state = self.env.reset() # Returns a tuple -> (ObsType, dict)
        self.frame_buffer = np.zeros((2, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.frame_buffer[0] = state[0]
        return state


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info
    

class ConvertToTorchTensor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation = torch.tensor(observation, dtype=torch.float)
        return observation
    

def make_env(env):
    env = MaxColor(env) # Finding max between two frames
    env = SkipFrame(env, skip=2) # skipping two frames (total for four frames processes so far)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)
    env = ConvertToTorchTensor(env)
    return env     
