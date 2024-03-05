from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import CosineSimilarityFunction


class KnowledgeDistillationKLLoss(nn.Module):
    def __init__(self, temperature: int) -> None:
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_scores: torch.FloatTensor | None = None,
        teacher_scores: torch.FloatTensor | None = None,
    ):
        soft_targets = nn.functional.softmax(
            teacher_scores / self.temperature, dim=-1,
        )
        soft_prob = nn.functional.log_softmax(
            student_scores / self.temperature, dim=-1,
        )

        kl_loss = self.kl_div(soft_prob, soft_targets) * (self.temperature**2)
        return kl_loss


class DistilKLLoss(nn.Module):
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

    def forward(
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
            teacher_pos_scores (torch.Tensor): The tensor of scores for positive examples as
            predicted by the teacher model.
            teacher_neg_scores (torch.Tensor): The tensor of scores for negative examples as
            predicted by the teacher model.

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
            [teacher_pos_scores, teacher_neg_scores], dim=1,
        )
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        return self.loss(local_scores, teacher_scores)


class DistilMarginMSE(nn.Module):
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

    def forward(
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
            teacher_pos_scores (torch.Tensor): The tensor of scores for positive examples
            as predicted by the teacher model.
            teacher_neg_scores (torch.Tensor): The tensor of scores for negative examples
            as predicted by the teacher model.

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


class Divergence(nn.Module):
    """
    A neural network module that computes the Jensen-Shannon (JS) divergence between two probability distributions.
    This divergence is a symmetric and smoothed version of the Kullback-Leibler (KL) divergence and is commonly
    used to measure the similarity between two probability distributions. In the context of neural networks,
    particularly for ranking tasks, this can be used to assess the consistency between rankings produced under
    different conditions, such as with different dropout masks.

    Attributes:
        kl (nn.KLDivLoss): The KL divergence loss with 'batchmean' reduction and log target enabled.
        eps (float): A small epsilon value to avoid log(0) which results in -inf.
        beta_ (float): A temperature parameter scaling the distributions, can control the smoothness
                       of the output distribution.

    Methods:
        forward(p, q): Computes the Jensen-Shannon divergence between two distributions.
    """

    def __init__(self, beta_):
        """
        Initializes the Divergence module with a specific beta value.

        Parameters:
            beta_ (float): The temperature parameter to scale the distributions.
        """
        super().__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.eps = 1e-7
        self.beta_ = beta_

    def forward(self, p: torch.tensor, q: torch.tensor) -> torch.Tensor:
        """
        Forward pass to compute the Jensen-Shannon divergence between two distributions.

        Parameters:
            p (torch.Tensor): The first probability distribution. Should be a 2D tensor where the
                              rows correspond to samples and the columns to class probabilities.
            q (torch.Tensor): The second probability distribution, in the same format as p.

        Returns:
            torch.Tensor: The Jensen-Shannon divergence between the two distributions. This is a single
                          scalar value if 'reduction' is set to 'batchmean' in the KL divergence.

        Note:
            The input distributions should be log-probabilities and they are normalized by this function
            before the JS divergence is computed. An epsilon value is used to clamp the log values to
            avoid taking the log of zero.
        """
        # Normalize the input distributions
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        # Compute the midpoint distribution between p and q
        m = (0.5 * (p + q)).clamp(min=self.eps)
        # Calculate the Jensen-Shannon divergence using the midpoint distribution and KL divergence
        return 0.5 * (self.kl(m, p) + self.kl(m, q))


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
    Implements the ListMLE loss function, a listwise learning to rank method. This function is used for
    ranking distillation by maximizing the likelihood of the correct ranking order (permutation) provided
    by a teacher model. It is particularly suited for training ranking models where the output
    needs to reflect a particular order.

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
