import numpy as np
import torch
from gym import ObservationWrapper, Wrapper
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation


class SkipFrame(Wrapper):
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


class ToTensor(ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def observation(self, obs):
        # Passing in a list of numpy arrays is slower than creating a tensor from a numpy array
        # Hence the `np.array(observation)` instead of `observation`
        # observation is a LIST of numpy arrays because of the LazyFrame wrapper
        return torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)


def apply_wrappers(env, device=None):
    env = SkipFrame(env, skip=4)  # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84)  # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(
        env, num_stack=4, lz4_compress=True
    )  # May need to change lz4_compress to False if issues arise

    if device is not None:
        # TODO: Am I negating the benefit of FrameStack by converting to tensor here?
        env = ToTensor(env, device)

    return env
