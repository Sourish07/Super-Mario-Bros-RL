import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.max_size = max_size
        self.mem_counter = 0

        self.state_memory = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.max_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_size, *input_shape), dtype=np.float32)
        self.done_memory = np.zeros(self.max_size, dtype=np.uint8)

    def add_transition(self, s, a, r, _s, done):
        index = self.mem_counter % self.max_size
        self.state_memory[index] = s
        self.action_memory[index] = a
        self.reward_memory[index] = r
        self.new_state_memory[index] = _s
        self.done_memory[index] = done

        self.mem_counter += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_counter, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, new_states, dones