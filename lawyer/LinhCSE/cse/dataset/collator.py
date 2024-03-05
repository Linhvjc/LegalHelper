from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


class AbstractDataCollator(ABC):
    """
    An abstract class for a PyTorch DataCollator. It defines the structure
    and the required method that must be implemented by any subclass.
    """

    @abstractmethod
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Abstract method that collates a batch of data samples into a format
        suitable for model training or evaluation.

        Parameters:
            batch (List[Dict[str, Any]]): A list of data samples where each sample
                                          is a dictionary mapping from string keys to data values.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping from string keys to torch.Tensors,
                                     collated from the input batch.
        """
        raise NotImplementedError('Subclasses should implement this method')
