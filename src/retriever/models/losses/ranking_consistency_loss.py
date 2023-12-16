from __future__ import annotations

import torch
import torch.nn as nn


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
        m = (0.5 * (p + q)).log().clamp(min=self.eps)
        # Calculate the Jensen-Shannon divergence using the midpoint distribution and KL divergence
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
