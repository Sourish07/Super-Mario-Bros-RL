import numpy as np
from pytorch_ddqn import DDQN
from replay_buffer import ReplayBuffer
import torch

class Agent:
    def __init__(self, input_dims, n_actions, lr=1e-4, gamma=0.99, epsilon=1, epsilon_end=0.1,
                 epsilon_decay=1e-5, mem_size=50000, batch_size=32,
                 copy_network_rate=10000, env_name="SuperMario", checkpoint_dir='checkpoints'):

        self.input_dims = input_dims
        self.n_actions = n_actions

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.copy_network_rate = copy_network_rate
        self.batch_size = batch_size

        self.buffer = ReplayBuffer(mem_size, input_dims)

        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir

        self.action_space = [a for a in range(self.n_actions)]

        self.learn_step_counter = 0

        self.online_network = DDQN(self.lr, self.n_actions, self.input_dims,
                                   f"{self.env_name}_online",
                                   self.checkpoint_dir)
        self.target_network = DDQN(self.lr, self.n_actions, self.input_dims,
                                   f"{self.env_name}_target",
                                   self.checkpoint_dir)

    def choose_action(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            # need to extra dimension because CNN expects first dimension to be of batch_size which is 1 in this case
            s = torch.tensor([s], dtype=torch.float).to(self.online_network.device)
            return torch.argmax(self.online_network.forward(s)).item()

    def store_transition(self, s, a, r, s_, done):
        self.buffer.add_transition(s, a, r, s_, done)

    def sample_memory(self):
        states, actions, rewards, new_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states).to(self.online_network.device) # Lowercase 'tensor' preserves the datatype of np
        # array
        actions = torch.tensor(actions).to(self.online_network.device)
        rewards = torch.tensor(rewards).to(self.online_network.device)
        new_states = torch.tensor(new_states).to(self.online_network.device)
        dones = torch.tensor(dones).to(self.online_network.device)

        return states, actions, rewards, new_states, dones

    def copy_network(self):
        if self.learn_step_counter % self.copy_network_rate == 0 and self.learn_step_counter > 0:
            print("Copying weights to target network")
            self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end

    def save_models(self, iter_num):
        print("...saving checkpoints...")
        self.online_network.save_checkpoint(iter_num)
        self.target_network.save_checkpoint(iter_num)

    def load_models(self, iter_num=None):
        print("...loading checkpoint...")
        self.online_network.load_checkpoint(iter_num)
        self.target_network.load_checkpoint(iter_num)

    def train(self):
        if self.buffer.mem_counter < self.batch_size:
            return

        self.online_network.optimizer.zero_grad()
        self.copy_network()

        states, actions, rewards, new_states, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        # Output shape will be batch_size x n_actions vvv
        q_pred = self.online_network.forward(states)[indices, actions]

        # max function returns (values, indices) so we need to take the first value only
        q_next = self.target_network.forward(new_states).max(dim=1)[0]

        q_next[dones.bool()] = 0.0 # Setting all terminal states to 0
        q_target = rewards + self.gamma * q_next

        loss = self.online_network.loss(q_target, q_pred).to(self.online_network.device)
        loss.backward()

        self.online_network.optimizer.step()
        self.decay_epsilon()

        self.learn_step_counter += 1




