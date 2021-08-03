import time
import numpy as np
import gym_super_mario_bros
import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from pytorch_agent import Agent
from wrappers import make_env

# Setting up environment
env_name = 'SuperMarioBros-1-1-v0'
env = gym_super_mario_bros.make(env_name)
env = JoypadSpace(env, RIGHT_ONLY)
env = make_env(env)

agent = Agent(env.observation_space.shape,
              env.action_space.n,
              env_name=env_name)

# Hyperparameters
num_of_episodes = 20000

scores, steps_array = [], []

load = False
if load:
    num_of_episodes = 10
    agent.load_models()
    agent.epsilon = 0.1

total_steps = 0

for e in range(num_of_episodes):
    start_time = time.time()
    state = env.reset()
    score = 0
    num_of_iterations = 0
    done = False

    while not done:
        env.render()
        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)

        if not load:
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

        score += reward
        state = next_state
        num_of_iterations += 1
        total_steps += 1

    if (e + 1) % 1000 == 0 and not load:
        agent.save_models(e + 1)

    scores.append(score)
    steps_array.append(total_steps)

    mean_reward = np.mean(scores[-100:])

    end_time = time.time()
    print(f'Episode #{e + 1}, '
          f'Score: {score}, '
          f'Number of Iterations: {num_of_iterations}, '
          f'Mean Reward: {mean_reward}, '
          f'Epsilon: {np.round(agent.epsilon, 4)}, '
          f'Time taken: {end_time - start_time}')

plt.plot(steps_array, scores)
plt.show()
