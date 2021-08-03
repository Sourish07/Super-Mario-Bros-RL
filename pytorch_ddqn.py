import os
import torch.nn as nn  # layers
import torch.nn.functional as F  # activation functions like relu or sigmoid
import torch.optim as optim  # optimizers
import torch as T
import numpy as np

class DDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name, checkpoint_dir):
        super(DDQN, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def calculate_conv_output(self, input_dims):
        x = T.zeros(1, *input_dims)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, s):
        s = F.relu(self.conv1(s))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s)) # conv3 shape is batch size x n_filters x H x W

        s = s.view(s.size()[0], -1) # batch_size by number of input features, kinda like numpy to reshape
        # ^ Flatten is so it's batch size (index value 0 in the dimensions tuple) by whatever

        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        return s

    def save_checkpoint(self, iter_num="_"):
        T.save(self.state_dict(), self.checkpoint_file)
        T.save(self.state_dict(), self.checkpoint_file + "_episode_" + str(iter_num))

    def load_checkpoint(self, iter_num=None):
        if iter_num is not None:
            checkpoint_file = self.checkpoint_file + "_episode_" + str(iter_num)
            self.load_state_dict(T.load(checkpoint_file))
        else:
            self.load_state_dict(T.load(self.checkpoint_file))

