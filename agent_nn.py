import torch
import numpy as np

class AgentNN(torch.nn.Module):
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
            torch.nn.Flatten(),
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )

        if freeze:
            self._freeze()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return self.network(x)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))
    
    def _freeze(self):        
        for p in self.network.parameters():
            p.requires_grad = False
    