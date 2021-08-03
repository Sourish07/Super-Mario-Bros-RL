import numpy as np
import gym
from gym import Wrapper, ObservationWrapper
from collections import deque
import cv2


class RepeatActionAndMaxFrame(Wrapper):
    def __init__(self, env, num_of_repeat, clip_rewards=False, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.num_of_repeat = num_of_repeat
        self.shape = env.observation_space.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        # self.clip_rewards = clip_rewards
        # self.no_ops = no_ops
        # self.fire_first = fire_first

    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self.num_of_repeat):
            obs, reward, done, info = self.env.step(action)
            #############################################
            # if self.clip_rewards:
            #     reward = np.clip(np.array([reward]), -1, 1).item()
            ###############################################
            total_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break
        #max_frame = self.frame_buffer.max(axis=0)
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        ####################################
        # Don't really understand this part
        # no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        # for _ in range(no_ops):
        #     _, _, done, _ = self.env.step(0)
        #     if done:
        #         self.env.reset()
        # if self.fire_first:
        #     assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
        #     obs, _, _, _, = self.env.step(1)
        #########################################
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs
        return obs




class PreprocessFrame(ObservationWrapper):
    def __init__(self, env, new_shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (new_shape[2], *new_shape[:2])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(observation, self.shape[1:], interpolation=cv2.INTER_AREA)
        observation = np.array(resized, dtype=np.uint8).reshape(self.shape)
        observation = np.swapaxes(observation, 2, 0)
        observation = observation / 255.
        return observation


class StackFrames(ObservationWrapper):
    def __init__(self, env, num_of_repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=env.observation_space.low.repeat(num_of_repeat, axis=0),
                                                high=env.observation_space.high.repeat(num_of_repeat, axis=0),
                                                dtype=np.float32)
        self.deque = deque(maxlen=num_of_repeat)

    def reset(self, **kwargs):
        self.deque.clear()
        obs = self.env.reset()
        for _ in range(self.deque.maxlen):
            self.deque.append(obs)
        return np.array(self.deque).reshape(self.observation_space.shape)

    def observation(self, observation):
        self.deque.append(observation)
        return np.array(self.deque).reshape(self.observation_space.shape)


def make_env(env, shape=(84, 84, 1), num_of_repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    #env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, num_of_repeat)
    #env = RepeatActionAndMaxFrame(env, num_of_repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, num_of_repeat)
    return env
