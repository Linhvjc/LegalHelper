from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class InferenceModel(ABC):
    """
    Abstract class for inference models.
    """

    @abstractmethod
    def __init__(self, model_path):
        """
        Constructor for initializing the inference model.

        Parameters:
        model_path (str): Path to the pre-trained model file.
        """
        pass

    @abstractmethod
    def load_model_and_tokenizer(self):
        """
        Abstract method to load the pre-trained model.
        """
        pass

    @abstractmethod
    def preprocess_input(self, input_data):
        """
        Abstract method to preprocess the input data before inference.

        Parameters:
        input_data: Raw input data that needs to be preprocessed.

        Returns:
        Preprocessed input data.
        """
        pass

    @abstractmethod
    def perform_inference(self, input_data):
        """
        Abstract method to perform inference using the pre-trained model.

        Parameters:
        input_data: Preprocessed input data for inference.

        Returns:
        Model predictions.
        """
        pass

    @abstractmethod
    def postprocess_output(self, output_data):
        """
        Abstract method to postprocess the output data after inference.

        Parameters:
        output_data: Model predictions that need to be postprocessed.

        Returns:
        Postprocessed output data.
        """
        pass

    def run_inference(self, input_data):
        """
        Method to run the complete inference pipeline.

        Parameters:
        input_data: Raw input data for inference.

        Returns:
        Processed output data.
        """
        preprocessed_data = self.preprocess_input(input_data)
        predictions = self.perform_inference(preprocessed_data)
        postprocessed_output = self.postprocess_output(predictions)
        return postprocessed_output
