from __future__ import annotations

import torch
import torch.nn as nn


class MaskedCrossEntropyLoss(nn.Module):
    """
    This class defines a custom cross-entropy loss function that incorporates an element-wise mask,
    allowing for selective consideration of elements during loss computation. Elements to be ignored
    are specified through the mask, enabling flexibility in loss calculation across different
    data samples and classes.
    """

    def __init__(self):
        """
        Constructs the MaskedCrossEntropyLoss module without any additional initialization parameters.
        Inherits from torch.nn.Module.
        """
        super().__init__()

    def forward(self, X: torch.Tensor, not_ignore: torch.Tensor) -> torch.Tensor:
        """
        Computes the masked cross-entropy loss given a batch of predictions and a corresponding mask.

        Parameters:
            X (torch.Tensor): Logits tensor with shape (batch_size, num_classes), containing raw,
                              unnormalized scores for each class.
            not_ignore (torch.Tensor): Mask tensor with shape (batch_size, num_classes), where
                              each element is 1 if it should contribute to the loss and 0 otherwise.

        Returns:
            torch.Tensor: Scalar tensor representing the mean cross-entropy loss over all non-ignored elements.
        """
        softmax = self.custom_softmax(X, not_ignore)
        # Epsilon added to avoid log(0)
        log_softmax = torch.log(softmax + 1e-10)
        return -torch.sum(log_softmax * not_ignore) / torch.sum(not_ignore)

    @staticmethod
    def custom_softmax(X: torch.Tensor, not_ignore: torch.Tensor) -> torch.Tensor:
        """
        Applies a masked softmax operation on the input logits, considering only the non-ignored elements.

        Parameters:
            X (torch.Tensor): Logits tensor with shape (batch_size, num_classes).
            not_ignore (torch.Tensor): Mask tensor with shape (batch_size, num_classes), determining
                              which elements to include in the softmax calculation.

        Returns:
            torch.Tensor: Softmax probabilities with the same shape as the input, computed only over non-ignored elements.
        """
        exp_x = torch.exp(X) * not_ignore
        sum_exp = torch.sum(exp_x, dim=-1, keepdim=True)
        softmax = exp_x / (sum_exp + 1e-10)  # Epsilon for numerical stability
        return softmax
