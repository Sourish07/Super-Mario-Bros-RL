import torch
import numpy as np

class AgentNeuralNetwork(torch.nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.network = torch.nn.Sequential(
            self.conv_layers,
            # start_dim default is 1 so it can ignore the batch dimension, 
            # but we don't have a batch dimension, so we set it to 0
            torch.nn.Flatten(start_dim=0),
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )

        if freeze:
            self.freeze()

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))
    
    def freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.network(x)
    