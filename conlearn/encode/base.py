from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from ..models.modules import Pooler


class DocumentEncoder(ABC):
    """
    Abstract base class for document encoding strategy.

    Attributes:
        pooler (Pooler): An instance of the Pooler class responsible for pooling operations.

    Args:
        pooler_type (str): The type of pooling strategy to use. Defaults to 'avg' for average pooling.
    """

    def __init__(self, pooler_type: str = 'avg') -> None:
        self.pooler = Pooler(pooler_type=pooler_type)

    @abstractmethod
    def encode(self, texts: list[str], **kwargs):
        """
        Abstract method for encoding a list of document texts.

        Args:
            texts (list[str]): A list of document texts to encode.

        Returns:
            The method should return an encoded representation of the provided texts.
        """
        pass


class QueryEncoder(ABC):
    """
    Abstract base class for query encoding strategy.

    Attributes:
        pooler (Pooler): An instance of the Pooler class responsible for pooling operations.

    Args:
        pooler_type (str): The type of pooling strategy to use. Defaults to 'avg' for average pooling.
    """

    def __init__(self, pooler_type: str = 'avg') -> None:
        self.pooler = Pooler(pooler_type=pooler_type)

    @abstractmethod
    def encode(self, texts: list[str], **kwargs):
        """
        Abstract method for encoding a list of query texts.

        Args:
            texts (list[str]): A list of query texts to encode.

        Returns:
            The method should return an encoded representation of the provided texts.
        """
        pass


class ModelEncoder(ABC):
    """
    Abstract base class for model-based encoding strategy.

    Attributes:
        pooler (Pooler): An instance of the Pooler class responsible for pooling operations.

    Args:
        pooler_type (str): The type of pooling strategy to use. Defaults to 'avg' for average pooling.
    """

    def __init__(self, pooler_type: str = 'avg') -> None:
        self.pooler = Pooler(pooler_type=pooler_type)

    @abstractmethod
    def encode(self, texts: list[str], max_seq_len: int, **kwargs):
        """
        Abstract method for encoding a list of texts with a maximum sequence length constraint.

        Args:
            texts (list[str]): A list of texts to encode.
            max_seq_len (int): The maximum sequence length for encoding.

        Returns:
            The method should return an encoded representation of the provided texts, considering the max sequence length constraint.
        """
        pass
