from __future__ import annotations

import torch


class DistilKLLoss:
    """
    Implements a knowledge distillation loss using the Kullback-Leibler (KL) divergence
    that measures how one probability distribution diverges from a second, expected probability distribution.

    Knowledge distillation involves a teacher-student training paradigm where the 'student' model
    is trained to mimic the 'teacher' model's behavior. This specific loss function computes the KL
    divergence between the student's predictions (a probability distribution across positive and
    negative scores) and the softened teacher's predictions, which helps in transferring the
    knowledge from the teacher to the student.

    The KL divergence is non-symmetric measure which means that the order of arguments
    (student scores and teacher scores) matters.

    Attributes:
        loss (torch.nn.KLDivLoss): The Kullback-Leibler divergence loss instance with no reduction.

    Methods:
        __call__(pos_scores, neg_scores, teacher_pos_scores, teacher_neg_scores): Computes the
        knowledge distillation loss given the student's and teacher's scores.
    """

    def __init__(self):
        """
        Initializes the DistilKLLoss class by setting up the KL divergence loss with no reduction
        which allows for manual control over how the loss should be aggregated.
        """
        self.loss = torch.nn.KLDivLoss(reduction='none')

    def __call__(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        teacher_pos_scores: torch.Tensor,
        teacher_neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the distillation KL divergence loss.

        Parameters:
            pos_scores (torch.Tensor): The tensor of scores for positive examples as predicted by the student model.
            neg_scores (torch.Tensor): The tensor of scores for negative examples as predicted by the student model.
            teacher_pos_scores (torch.Tensor): The tensor of scores for positive examples as predicted by the teacher model.
            teacher_neg_scores (torch.Tensor): The tensor of scores for negative examples as predicted by the teacher model.

        Returns:
            torch.Tensor: The computed KL divergence loss as a scalar tensor.
        """
        # Concatenate the positive and negative scores for both student and teacher
        # and compute the softmax probabilities for the teacher's scores. The student's
        # scores are log-softmaxed to ensure numerical stability when passed to the
        # KL divergence loss function. Finally, the loss is computed, summed over the
        # positive and negative examples, and then the mean is taken.
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        local_scores = torch.log_softmax(scores, dim=1)
        teacher_scores = torch.cat(
            [teacher_pos_scores.unsqueeze(-1), teacher_neg_scores.unsqueeze(-1)], dim=1,
        )
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        return self.loss(local_scores, teacher_scores).sum(dim=1).mean(dim=0)
