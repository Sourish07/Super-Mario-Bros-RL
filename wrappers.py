import gym
import numpy as np

# Trying to find the max color between two consecutive frames due to flickering
class MaxColor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frame_buffer = np.zeros((2, *self.observation_space.shape))

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
        self.frame_buffer = np.zeros((2, *self.observation_space.shape))
        self.frame_buffer[0] = state[0]
        return state
