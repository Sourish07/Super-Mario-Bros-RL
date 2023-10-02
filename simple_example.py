import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
# import matplotlib.pyplot as plt


from nes_py.wrappers import JoypadSpace

ENV_NAME = 'SuperMarioBros-1-1-v0'

# env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

done = False
env.reset()
counter = 0
while not done:
    # frame = env.render()
    # plt.imsave(f"frames/frame{counter}.png", frame)
    # counter += 1
    # Choose right only action
    action = RIGHT_ONLY.index(['right'])

    _, _, done, _, _ = env.step(action)
    env.render()