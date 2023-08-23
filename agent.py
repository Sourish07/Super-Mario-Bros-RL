import torch
import numpy as np
from agent_nn import AgentNN

class Agent:
    def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99, epsilon=1.0, eps_decay=1e-5, eps_min=0.1):
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

        # self.replay_buffer = 


    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        observation = torch.tensor(observation, dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        return self.online_network(observation)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        pass

    def learn(self):
        pass