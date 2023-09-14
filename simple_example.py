import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY


from nes_py.wrappers import JoypadSpace

ENV_NAME = 'SuperMarioBros-1-1-v0'

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

done = False
state = env.reset()
while not done:
    _, _, done, _, _ = env.step(env.action_space.sample())
    env.render()