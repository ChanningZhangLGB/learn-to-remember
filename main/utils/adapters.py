"""
Adapter modules for lightweight online learning.
"""

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """
    Small bottleneck MLP adapter.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
