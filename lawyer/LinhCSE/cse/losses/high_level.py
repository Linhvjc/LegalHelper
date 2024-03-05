from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            'forward method must be implemented in the subclass.',
        )


class PairwiseBPR(PairwiseLoss):
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
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
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


class PairwiseNLL(PairwiseLoss):
    """
    Implements the pairwise negative log-likelihood loss for ranking tasks.

    This loss function takes a batch of scores for positive and negative examples
    and applies a log-softmax to them, treating the first column as the positive
    scores and the remaining as negative. The loss is the mean negative log-likelihood
    of the positive scores.

    Attributes:
        logsoftmax (torch.nn.LogSoftmax): The log-softmax function applied across the scores.

    Methods:
        __call__(pos_scores, neg_scores): Computes the loss given positive and negative scores.
    """

    def __init__(self):
        super().__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute the Pairwise NLL loss.

        Parameters:
            pos_scores (torch.Tensor): The tensor of scores for positive examples.
            neg_scores (torch.Tensor): The tensor of scores for negative examples.

        Returns:
            torch.Tensor: The computed Pairwise NLL loss.
        """
        scores = self.logsoftmax(torch.cat([pos_scores, neg_scores], dim=1))
        return torch.mean(-scores[:, 0])


class InBatchPairwiseNLL(PairwiseLoss):
    """
    Implements the in-batch pairwise negative log-likelihood loss.

    This version of the loss is designed for scenarios where negative samples
    are taken from the same batch (in-batch negatives). The scores for the positive
    examples and negative examples (from BM25 sampling or other methods) are combined
    and passed through a log-softmax. The loss is the mean negative log-likelihood
    over the modified scores, which is designed to work with distributed settings
    where the batch is split across multiple GPUs.

    Attributes:
        logsoftmax (torch.nn.LogSoftmax): The log-softmax function applied to the scores.

    Methods:
        __call__(in_batch_scores, neg_scores): Computes the loss using in-batch scores and additional negative scores.
    """

    def __init__(self):
        super().__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, in_batch_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute the In-Batch Pairwise NLL loss.

        Parameters:
            in_batch_scores (torch.Tensor): The tensor of scores within the batch.
                Expected to be a square matrix if not distributed, or a matrix of size
                (batch_size * (batch_size / num_gpus)) in distributed settings.
            neg_scores (torch.Tensor): The tensor of additional negative scores.

        Returns:
            torch.Tensor: The computed In-Batch Pairwise NLL loss.
        """
        # The number of columns corresponds to the number of positive examples per GPU
        nb_columns = in_batch_scores.shape[1]
        # The number of GPUs is inferred from the batch size and number of columns
        nb_gpus = int(in_batch_scores.shape[0] / nb_columns)
        temp = torch.cat([in_batch_scores, neg_scores], dim=1)
        scores = self.logsoftmax(temp)
        # We select the scores corresponding to the positive examples for the loss calculation
        return torch.mean(
            -scores[
                torch.arange(in_batch_scores.shape[0]), torch.arange(
                    nb_columns,
                ).repeat(nb_gpus),
            ],
        )
