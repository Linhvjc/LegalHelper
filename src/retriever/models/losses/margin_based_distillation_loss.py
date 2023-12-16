from __future__ import annotations

import torch


class DistilMarginMSE:
    """
    Implements a margin-based Mean Squared Error (MSE) distillation loss, specifically designed for knowledge
    distillation across different neural ranking model architectures. This loss function is inspired by the
    approach described in "Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation"
    (https://arxiv.org/abs/2010.02666).

    The objective of this loss function is to minimize the squared differences between the margins (positive score
    minus negative score) of the student model and the teacher model, effectively forcing the student to mimic the
    margin distribution of the teacher.

    Attributes:
        loss (torch.nn.MSELoss): The mean squared error loss instance.

    Methods:
        __call__(pos_scores, neg_scores, teacher_pos_scores, teacher_neg_scores): Computes the
        margin-based MSE distillation loss given the scores from the student and teacher models.
    """

    def __init__(self):
        """
        Initializes the DistilMarginMSE class by setting up the mean squared error loss.
        """
        self.loss = torch.nn.MSELoss()

    def __call__(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        teacher_pos_scores: torch.Tensor,
        teacher_neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the margin-based MSE distillation loss.

        Parameters:
            pos_scores (torch.Tensor): The tensor of scores for positive examples as predicted by the student model.
            neg_scores (torch.Tensor): The tensor of scores for negative examples as predicted by the student model.
            teacher_pos_scores (torch.Tensor): The tensor of scores for positive examples as predicted by the teacher model.
            teacher_neg_scores (torch.Tensor): The tensor of scores for negative examples as predicted by the teacher model.

        Returns:
            torch.Tensor: The computed MSE loss as a scalar tensor.

        Note:
            This function assumes that the input tensors contain raw scores from the respective models,
            where 'pos_scores' and 'teacher_pos_scores' correspond to positive instances and 'neg_scores'
            and 'teacher_neg_scores' correspond to negative instances. The loss computation squeezes
            the margins to ensure that the dimensions are consistent for MSE calculation.
        """
        # Calculate the margins for both the student and teacher predictions.
        # The margin is the difference between the positive and negative scores.
        # The squeeze() function is used to remove dimensions of size 1, which
        # may be present after the subtraction operation. The MSE loss is then
        # applied to the student-teacher margin pairs to enforce similarity.
        margin = pos_scores - neg_scores
        teacher_margin = teacher_pos_scores - teacher_neg_scores
        return self.loss(margin.squeeze(), teacher_margin.squeeze())
