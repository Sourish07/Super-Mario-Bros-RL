import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from agent_nn import AgentNN


class Agent:
    def __init__(
        self,
        input_dims,
        num_actions,
        lr=0.00025,
        gamma=0.9,
        epsilon=1.0,
        eps_decay=0.99999975,
        eps_min=0.1,
        replay_buffer_capacity=100_000,
        batch_size=32,
        sync_network_rate=10000,
    ):
        self.num_actions = num_actions
        self.learn_step_counter = 0

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        # Networks
        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.SmoothL1Loss() # Feel free to try this loss function instead!

        # Replay buffer
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def choose_action(self, observation, epsilon_greedy=True):
        # If epsilon_greedy is False, then the agent will always choose the action with the highest Q-value
        if epsilon_greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = observation.unsqueeze(0)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            TensorDict(
                {
                    "state": state,
                    "action": torch.tensor(action),
                    "reward": torch.tensor(reward),
                    "next_state": next_state,
                    "done": torch.tensor(done),
                },
                batch_size=[],
            )
        )

    def sync_networks(self):
        if (
            self.learn_step_counter % self.sync_network_rate == 0
            and self.learn_step_counter > 0
        ):
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(torch.load(path, map_location=self.device))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()

        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_network(
            states
        )  # Shape is (batch_size, n_actions)
        predicted_q_values = predicted_q_values[
            np.arange(self.batch_size), actions.squeeze()
        ]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        # The rewards of any future states don't matter if the current state is a terminal state
        # If done is true, then 1 - done is 0, so the part after the plus sign (representing the future rewards) is 0
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

    def to(self, device):
        self.device = device
        self.online_network.to(device)
        self.target_network.to(device)
        return self
