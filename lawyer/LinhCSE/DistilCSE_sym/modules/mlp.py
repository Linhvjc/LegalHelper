from __future__ import annotations

import torch.nn as nn


class MLPLayer(nn.Module):
    """
    Head for obtaining sentence representations over transformers' representation.
    Configures an MLP with a specific architecture, including residual connections.
    """

    def __init__(self):
        super().__init__()
        # Use a list to store layers if you have a fixed number of layers
        modules = [
            nn.Linear(1024, 1024),
            nn.Linear(1024, 768),
            nn.Linear(768, 768),
            ]
        self.layers = nn.ModuleList(modules)

        # Use a dictionary to store activation functions if you have a fixed pattern
        self.activations = {0: nn.GELU(), 1: nn.GELU(), 2: nn.ReLU()}

    def forward(self, features, **kwargs):
        x = features
        for i, layer in enumerate(self.layers):
            x = self.activations[i](layer(x))
        return (
            x
            # features + x
        )  # Assuming you want to add the input features as a residual connection to the output
