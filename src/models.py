# manufacturing_rl/models.py
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 256, num_layers: int = 3) -> None:
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x