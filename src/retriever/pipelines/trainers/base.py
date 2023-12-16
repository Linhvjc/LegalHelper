from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class Trainer(ABC):
    """
    Abstract class for training models.
    """

    @abstractmethod
    def __init__(self, model_path: str):
        """
        Constructor for initializing the trainer with a model.

        Parameters:
        model_path (str): Path to the pre-trained model file.
        """
        pass

    @abstractmethod
    def train(self, training_data):
        """
        Abstract method to train the model on the provided data.

        Parameters:
        training_data: The data on which the model should be trained. The type of this parameter
                       depends on the specific implementation (could be a path to a data file,
                       in-memory data structure, etc.).
        """
        pass

    @abstractmethod
    def evaluate(self, input_data):
        """
        Abstract method to evaluate the trained model.

        Parameters:
        input_data: The input data to evaluate the model. The type of this parameter
                    depends on the specific implementation.
        """
        pass
