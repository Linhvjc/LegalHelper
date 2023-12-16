from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityFunctionWithTorch(nn.Module):
    """
    Module for computing similarity scores between two sets of vectors using either
    dot product or cosine similarity.
    """

    def __init__(self, name_fn='cosine'):
        """
        Initializes the SimilarityFunctionWithTorch module.

        Parameters:
        name_fn (str): The name of the similarity function to use.
                       Accepts 'dot' for dot product or 'cosine' for cosine similarity.
        """
        super().__init__()
        if name_fn not in ['dot', 'cosine']:
            raise ValueError(
                "Invalid value for name_fn. Supported values are 'cosine' and 'dot'.",
            )
        self.name_fn = name_fn

    def forward(self, compr: torch.Tensor, refer: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity scores between two sets of vectors.

        Parameters:
        compr (torch.Tensor): A tensor of shape (batch_size, features).
        refer (torch.Tensor): A tensor of shape (batch_size, features).

        Returns:
        torch.Tensor: A tensor of similarity scores.
        """
        if self.name_fn == 'dot':
            return self.dot_product_distance(compr, refer)
        elif self.name_fn == 'cosine':
            return self.cosine_distance(compr, refer)

    @staticmethod
    def dot_product_distance(compr: torch.Tensor, refer: torch.Tensor) -> torch.Tensor:
        """
        Computes the dot product scores between two sets of vectors.

        Parameters:
        compr (torch.Tensor): A tensor of shape (batch_size, features).
        refer (torch.Tensor): A tensor of shape (batch_size, features).

        Returns:
        torch.Tensor: A tensor of dot product scores.
        """
        refer_t = torch.transpose(refer, 0, 1)
        return torch.matmul(compr, refer_t)

    @staticmethod
    def cosine_distance(compr: torch.Tensor, refer: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity scores between two sets of vectors.

        Parameters:
        compr (torch.Tensor): A tensor of shape (batch_size, features).
        refer (torch.Tensor): A tensor of shape (batch_size, features).

        Returns:
        torch.Tensor: A tensor of cosine similarity scores.
        """
        # Unsqueeze 'refer' to add batch dimension for cosine_similarity
        compr_unsqueezed = compr.unsqueeze(0)  # (batch_size, 1, features)
        refer_unsqueezed = refer.unsqueeze(1)  # (1, batch_size, features).
        return F.cosine_similarity(compr_unsqueezed, refer_unsqueezed, dim=-1)


class SimilarityFunctionWithNumpy:
    """
    Class for computing similarity scores between two sets of vectors using Numpy.
    Provides options for either dot product or cosine similarity calculations.
    """

    def __init__(self, method='cosine'):
        """
        Initializes the SimilarityFunctionNumpy with the specified method.

        Parameters:
        method (str): The similarity calculation method. Either 'dot' or 'cosine'.
        """
        valid_methods = ['dot', 'cosine']
        if method not in valid_methods:
            raise ValueError(
                f'Invalid method. Supported methods are {valid_methods}',
            )
        self.method = method

    def compute_scores(self, compr, refer):
        """
        Computes the similarity scores between two sets of vectors.

        Parameters:
        compr (numpy.ndarray): A 2D array of shape (batch_size, features).
        refer (numpy.ndarray): A 2D array of shape (features, sequence_length).

        Returns:
        numpy.ndarray: An array of similarity scores.
        """
        if self.method == 'dot':
            return self.dot_product_scores(compr, refer)
        elif self.method == 'cosine':
            return self.cosine_scores(compr, refer)

    @staticmethod
    def dot_product_scores(compr, refer):
        """
        Computes the dot product scores between two sets of vectors using Numpy.

        Parameters:
        compr (numpy.ndarray): A 2D array of shape (batch_size, features).
        refer (numpy.ndarray): A 2D array of shape (batch_size, features).

        Returns:
        numpy.ndarray: An array of dot product scores.
        """
        return np.matmul(compr, refer.T)

    @staticmethod
    def cosine_scores(compr, refer):
        """
        Computes the cosine similarity scores between two sets of vectors using Numpy.

        Parameters:
        compr (numpy.ndarray): A 2D array of shape (batch_size, features).
        refer (numpy.ndarray): A 2D array of shape (sequence_length, features).

        Returns:
        numpy.ndarray: An array of cosine similarity scores.
        """
        compr_norm = np.linalg.norm(compr, axis=1, keepdims=True)
        refer_norm = np.linalg.norm(refer, axis=1)
        dot_product = np.dot(compr, refer.T)
        similarity = dot_product / (compr_norm * refer_norm[np.newaxis, :])
        return similarity
