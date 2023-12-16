from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ListNet(nn.Module):
    """
    Implements the ListNet ranking distillation objective, which minimizes the cross-entropy
    between the predicted permutation probability distribution and the ground truth distribution
    from the teacher.

    Attributes:
        teacher_temp_scaled_sim (CosineSimilarityFunction): Computes the cosine similarity with
                                                           half the tau value as temperature.
        student_temp_scaled_sim (CosineSimilarityFunction): Computes the cosine similarity with
                                                           tau as temperature.
        gamma_ (float): A scaling factor for the loss.

    Methods:
        forward(teacher_top1_sim_pred, student_top1_sim_pred): Computes the ListNet loss.
    """

    def __init__(self, tau: float, gamma_: float):
        """
        Initializes the ListNet module.

        Parameters:
            tau (float): Temperature scaling factor for the cosine similarity.
            gamma_ (float): Scaling factor for the loss.
        """
        super().__init__()
        self.teacher_temp_scaled_sim = CosineSimilarityFunction(tau / 2)
        self.student_temp_scaled_sim = CosineSimilarityFunction(tau)
        self.gamma_ = gamma_

    def forward(
        self, teacher_top1_sim_pred: torch.Tensor, student_top1_sim_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the ListNet loss given the teacher and student predictions.

        Parameters:
            teacher_top1_sim_pred (torch.Tensor): Teacher's predictions.
            student_top1_sim_pred (torch.Tensor): Student's predictions.

        Returns:
            torch.Tensor: The ListNet loss.
        """
        p = F.log_softmax(
            student_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1,
        )
        q = F.softmax(
            teacher_top1_sim_pred.fill_diagonal_(
                float('-inf'),
            ), dim=-1,
        )
        loss = -(q * p).nansum() / q.nansum()
        return self.gamma_ * loss


class ListMLE(nn.Module):
    """
    Implements the ListMLE loss function, a listwise learning to rank method. This function is used for ranking distillation by maximizing the likelihood of the correct ranking order (permutation) provided by a teacher model. It is particularly suited for training ranking models where the output needs to reflect a particular order.

    Attributes:
        temp_scaled_sim (CosineSimilarityFunction): An instance of `CosineSimilarityFunction`
            for calculating temperature-scaled cosine similarity between vectors.
        gamma_ (float): A factor that scales the computed loss value.
        eps (float): A small epsilon value for numerical stability during computations.

    Methods:
        forward(teacher_top1_sim_pred, student_top1_sim_pred): Computes the ListMLE loss between
            the predicted similarities from the student and the teacher models.
    """

    def __init__(self, tau: float, gamma_: float):
        """
        Initializes the ListMLE module with a specified temperature for similarity scaling and a
        loss scaling factor.

        Parameters:
            tau (float): The temperature value used to scale the cosine similarity computation.
            gamma_ (float): The scaling factor for the final loss value.
        """
        super().__init__()
        self.temp_scaled_sim = CosineSimilarityFunction(tau)
        self.gamma_ = gamma_
        self.eps = 1e-7

    def forward(
        self, teacher_top1_sim_pred: torch.Tensor, student_top1_sim_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the ListMLE loss computation. It calculates the loss based on the
        predicted ranking similarity scores from the student model and the target scores provided
        by the teacher model.

        The method involves shuffling for randomized tie resolution, sorting based on true values,
        and computing a cumulative loss for all items in the ranking.

        Parameters:
            teacher_top1_sim_pred (torch.Tensor): The tensor of top-1 similarity scores predicted
                by the teacher model.
            student_top1_sim_pred (torch.Tensor): The tensor of top-1 similarity scores predicted
                by the student model.

        Returns:
            torch.Tensor: A scalar tensor representing the mean ListMLE loss across the batch.
        """
        y_pred = student_top1_sim_pred
        y_true = teacher_top1_sim_pred

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(
            y_pred_shuffled, dim=1, index=indices,
        )
        preds_sorted_by_true[mask] = float('-inf')
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(
            dims=[1],
        )
        observation_loss = torch.log(
            cumsums + self.eps,
        ) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return self.gamma_ * torch.mean(torch.sum(observation_loss, dim=1))
