from __future__ import annotations

import torch


class PairwiseBPR:
    """
    Implements the Bayesian Personalized Ranking (BPR) loss function for implicit feedback.

    BPR is a pairwise loss function that is often used in recommendation systems where
    the goal is to learn from user interactions that are only positive (implicit feedback),
    such as purchases, clicks, or views, without corresponding negative interactions.

    The loss is calculated as the binary cross-entropy between the predicted preference
    of a user for a positive item over a negative item and a target of ones, which
    represents the assumption that users prefer positive items over negative ones.

    Attributes:
        device (torch.device): The device to run the computations on, either 'cuda' for GPU
            or 'cpu' for CPU based on CUDA availability.
        loss (torch.nn.BCEWithLogitsLoss): The binary cross-entropy loss with logits
            function to compute the BPR loss.

    Methods:
        __call__(pos_scores, neg_scores): Computes the BPR loss given scores of
            positive and negative items.
    """

    def __init__(self):
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def __call__(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute the BPR loss given the positive and negative item scores.

        Parameters:
            pos_scores (torch.Tensor): The tensor of scores for positive examples (items).
            neg_scores (torch.Tensor): The tensor of scores for negative examples (items).

        Returns:
            torch.Tensor: The computed BPR loss as a scalar tensor.
        """
        # The positive and negative item scores are subtracted, and the result is
        # squeezed to remove any dimensions of size 1. A target tensor of ones with the
        # same length as the positive scores is created on the same device. The BPR loss
        # is then computed as the binary cross-entropy with logits.
        return self.loss(
            (
                pos_scores -
                neg_scores
            ).squeeze(), torch.ones(pos_scores.shape[0]),
        )
