import os
import torch
import torch.nn as nn

class ResNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ResNetMLP, self).__init__()
        self.num_layers = num_layers
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.residual_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_in(x))
        residual = x
        for i in range(self.num_layers):
            out = self.relu(self.residual_layers[i](x))
            x = out + residual  # ResNet skip connection
            residual = x
        x = self.fc_out(x)
        return x