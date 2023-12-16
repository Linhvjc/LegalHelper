from __future__ import annotations

import torch


class PairwiseNLL:
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
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
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


class InBatchPairwiseNLL:
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
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, in_batch_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
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
