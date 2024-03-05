from __future__ import annotations

import torch.nn as nn


class CosineSimilarityFunction(nn.Module):
    """
    Computes the cosine similarity between two tensors. Optionally, the cosine similarity
    can be divided by a temperature scaling factor to soften the similarity values.

    Attributes:
        temp (float): Temperature scaling factor to divide the cosine similarity.
        cos (nn.CosineSimilarity): Module to compute cosine similarity along the last dimension.

    Methods:
        forward(x, y): Returns the cosine similarity between two tensors x and y divided by the temperature.
    """

    def __init__(self, temp):
        """
        Initializes the CosineSimilarityFunction with a specified temperature.

        Parameters:
            temp (float): Temperature scaling factor.
        """
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        """
        Computes the cosine similarity between two tensors and divides the result by the temperature.

        Parameters:
            x (torch.Tensor): The first tensor.
            y (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The cosine similarity between x and y divided by the temperature.
        """
        return self.cos(x, y) / self.temp
