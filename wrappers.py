import gym
import numpy as np
import torch
import torchvision
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
    

class MyGrayscaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        obs_shape = self.observation_space.shape[:2]
        # self.observation_space.low_repr is the string representation of minimum value in observation space
        # self.observation_space.low is the array of minimum values of the observation space (it's a bunch of zeros) of shape (240, 256, 3)
        self.observation_space = gym.spaces.Box(low=np.min(self.observation_space.low),
                                                high=np.min(self.observation_space.high),
                                                shape=obs_shape,
                                                dtype=self.observation_space.dtype)
        
    def observation(self, observation):
        # Rearranging the dimensions of the observation from [H, W, C] to [C, H, W]
        # PyTorch Grayscale expects the input to be [..., 3, H, W]
        observation = np.transpose(observation, (2, 0, 1))
        # Converting np array to torch tensor
        observation = torch.tensor(observation.copy(), dtype=torch.float)

        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        print("In grayscale, shape after transform:", observation.shape)
        return observation
    

class MyResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        # Replacing the height and the width of the observation space with the new shape
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape) # tuple constructor takes in an iterable so the tuple won't have the list or tuple inside it
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=np.min(self.observation_space.low),
                                                high=np.min(self.observation_space.high),
                                                shape=obs_shape,
                                                dtype=self.observation_space.dtype)
        
    def observation(self, observation):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.shape),
            torchvision.transforms.Normalize(0, 255)
        ])
        observation = transforms(observation)
        # Shape after resize is [1, 84, 84]
        observation = observation.squeeze(0) # Removes the first dimension of the observation if it is 1
        # Now shape is [84, 84]
        return observation
    

class ConvertToTorchTensor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        observation = torch.tensor(observation, dtype=torch.float)
        return observation
            

class MyFrameStack(gym.ObservationWrapper):
    def __init__(self, env, num):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=np.min(self.observation_space.low),
                                                high=np.min(self.observation_space.high),
                                                shape=(self.observation_space.shape[0] * num, *self.observation_space.shape[1:]),
                                                dtype=self.observation_space.dtype)
        self.frames = deque(maxlen=num)
    

def make_env(env):
    env = MaxColor(env) # Finding max between two frames
    env = SkipFrame(env, skip=2) # skipping two frames (total for four frames processes so far)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)
    env = ConvertToTorchTensor(env)
    return env     
