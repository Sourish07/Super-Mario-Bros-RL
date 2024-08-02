import os

from agent import Agent
from utils import *

device = get_device()
dummy_env = make_env()

agent = Agent(
    input_dims=dummy_env.observation_space.shape, num_actions=dummy_env.action_space.n
)
agent.to(device)

folder_name = ""
ckpt_name = ""
agent.load_model(os.path.join("models", folder_name, ckpt_name), device=device)

total_reward, flag_reached = eval_run(agent, device)
print(f"Total reward: {total_reward}, Flag reached: {flag_reached}")
