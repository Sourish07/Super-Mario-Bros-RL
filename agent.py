import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage, ListStorage


class Agent:
    def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99, epsilon=1.0, eps_decay=1e-5, eps_min=0.1, replay_buffer_capacity=100_000, batch_size=32):
        self.action_space = [i for i in range(n_actions)]
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        print("input_dims", input_dims)
        self.online_network = AgentNN(input_dims, n_actions)
        self.target_network = AgentNN(input_dims, n_actions, freeze=True)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.SmoothL1Loss()
        # self.loss = torch.nn.MSELoss()

        storage = ListStorage(replay_buffer_capacity) # , device=self.online_network.device
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)
        self.batch_size = batch_size

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        # Passing in a list of numpy arrays is slower than creating a tensor from a numpy array
        # Hence the `observation.__array__()` instead of `observation`
        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = torch.tensor(observation.__array__(), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({"state": state.__array__(), "action": action, "reward": reward, "next_state": next_state.__array__(), "done": done}, batch_size=[]))

    def learn(self):
        pass