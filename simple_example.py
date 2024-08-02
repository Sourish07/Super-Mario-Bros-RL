from utils import make_env

env = make_env(display=True)

done = False
env.reset()
counter = 0
while not done:
    # Only go right
    # action = RIGHT_ONLY.index(['right'])

    # Choose random action
    action = env.action_space.sample()

    _, _, done, _, _ = env.step(action)
    env.render()

env.close()
