import os

from agent import Agent
from utils import *

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

set_default_logger(model_path)

device = get_device()

CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000

env = make_env(display=False, device=device)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
agent.to(device)

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, _, info = env.step(a)

        agent.store_in_memory(state, a, reward, new_state, done)
        agent.learn()

        state = new_state

    total_reward, _ = eval_run(agent, device=device)
    logging.info(
        "Episode: %d, Total reward: %d, Agent learn step counter: %d, Epsilon: %f, Replay buffer size: %d",
        i,
        total_reward,
        agent.learn_step_counter,
        agent.epsilon,
        len(agent.replay_buffer),
    )

    if (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

env.close()
