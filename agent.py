import torch
import numpy as np
from agent_nn import AgentNN

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, LazyTensorStorage, ListStorage


class Agent:
    def __init__(self, input_dims, n_actions, lr=0.0001, gamma=0.99, epsilon=1.0, eps_decay=1e-5, eps_min=0.1, replay_buffer_capacity=100_000, batch_size=32, sync_network_rate=10000):
        self.action_space = [i for i in range(n_actions)]
        
        self.learn_step_counter = 0
        self.sync_network_rate = sync_network_rate

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
        # observation is a list of numpy arrays because of the LazyFrame wrapper
        # Unqueeze adds a dimension to the tensor, which represents the batch dimension
        observation = torch.tensor(observation.__array__(), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        # Grabbing the index of the action that's associated with the highest Q-value
        return self.online_network(observation).argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(state.__array__(), dtype=torch.float32).to(self.online_network.device), 
                                            "action": torch.tensor(action).to(self.online_network.device),
                                            "reward": torch.tensor(reward).to(self.online_network.device), 
                                            "next_state": torch.tensor(next_state.__array__(), dtype=torch.float32).to(self.online_network.device), 
                                            "done": torch.tensor(done).to(self.online_network.device)
                                          }, batch_size=[]))
        
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()
        if self.learn_step_counter % 10000 == 0 and self.learn_step_counter > 0:
            self.save_model(f"models/model_{self.learn_step_counter}_iter.pt")

        if self.learn_step_counter % 1000 == 0 and self.learn_step_counter > 0:
            print("Size of replay buffer:", len(self.replay_buffer))
        
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = [samples[key] for key in ("state", "action", "reward", "next_state", "done")]

        predicted_rewards = self.online_network(states) # Shape is (batch_size, n_actions)
        predicted_rewards = predicted_rewards[np.arange(self.batch_size), actions]

        # Max returns two tensors, the first one is the maximum value, the second one is the index of the maximum value
        target_rewards = self.target_network(next_states).max(dim=1)[0]
        # The rewards of any future states don't matter if the current state is a terminal state
        target_rewards = rewards + self.gamma * target_rewards * ~dones

        loss = self.loss(predicted_rewards, target_rewards)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()


        


