import datetime
import logging
import os

import gym_super_mario_bros
import torch
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from wrappers import apply_wrappers


def make_env(display=False, device=None):
    ENV_NAME = "SuperMarioBros-1-1-v0"
    env = gym_super_mario_bros.make(
        ENV_NAME,
        render_mode="human" if display else "rgb",
        apply_api_compatibility=True,
    )
    env = JoypadSpace(env, RIGHT_ONLY)

    return apply_wrappers(env, device)


def eval_run(agent, device):
    env = make_env(display=False, device=device)
    with torch.no_grad():
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state, epsilon_greedy=False)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            total_reward += reward

        env.close()
    return total_reward, info["flag_get"]


def set_default_logger(path=""):
    logging.basicConfig(
        filename=os.path.join(path, "training.log"),
        filemode="w",  # 'w' will overwrite the file each time the application is run
        level=logging.DEBUG,  # Capture all messages of DEBUG and above
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_current_date_time_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def get_device():
    if torch.cuda.is_available():
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # CPU might be faster for some Mac users
        # device = torch.device("cpu")
        logging.info("Using device: Apple Silicon GPU")
        return torch.device("mps")
    else:
        logging.info("Using device: CPU")
        return torch.device("cpu")
