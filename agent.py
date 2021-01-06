from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np

from collections import deque
import random


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.batch_size = 32
        self.learning_rate = 0.000025
        self.model = self.build_model()

        self.model_file = 'model.h5'

    def build_model(self):
        i = Input(shape=self.state_size)
        x = Conv2D(32, (8, 8), strides=4, activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(self.action_size, name='output')(x)

        model = Model(i, x)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(low=0, high=self.action_size)
            # print(f'Random action taken: {SIMPLE_MOVEMENT[action]}')
        else:
            actions = self.model.predict(np.expand_dims(state, 0))
            action = np.argmax(actions)
            # print(f'Calculated action taken: {SIMPLE_MOVEMENT[action]}')
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # print('Training model!')
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))

        q = self.model.predict(state)
        q_next = self.model.predict(next_state)

        batch_index = np.arange(self.batch_size)

        q[batch_index, action] = reward + (1. - done) * self.gamma * np.amax(q_next, axis=1)

        self.model.train_on_batch(state, q)

        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        # print(f'New epsilon: {self.epsilon}')

    def save_model(self, iteration_number):
        self.model.save(f'checkpoints/iteration-{iteration_number}.h5')
        self.model.save(self.model_file)

    def load_model(self):
        self.model = load_model(f'{self.model_file}')
