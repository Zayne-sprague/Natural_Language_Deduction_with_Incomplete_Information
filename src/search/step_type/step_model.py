import torch
from typing import List, Union
import transformers

from step_model.inference import get_model, get_tokenizer
from search.tree.step_generation import StepGeneration
from search.tree.tree import Tree


class StepModel:
    """StepModel base class that stores the torch model, tokenizer, and device to be used for sampling output."""

    model: torch.nn.Module
    tokenizer: transformers.AutoTokenizer
    device: torch.device

    def __init__(
            self,
            model_name: str,
            max_output_length: int = 64,
            num_return_sequences: int = 1,
            batch_size: int = 4,
            device: torch.device = torch.device('cpu'),
            force_new_instance: bool = False
    ):
        """
        :param model_name: The name of the model (name of the folder in trained_models)
        :param max_output_length: Integer length of tokens returned by the step model
        :param device: Torch device to run the model on
        :param force_new_instance: This class will try to use a signleton pattern so that you do not load the same model
            twice, you can skip this logic by turning this parameter to true.
        """

        if hasattr(self, 'instantiated'):
            return

        self.device = torch.device(device)
        self.model = get_model(model_name, max_length=max_output_length, device=self.device)
        self.tokenizer = get_tokenizer(model_name)
        self.num_return_sequences = num_return_sequences
        self.batch_size = batch_size
        self.instantiated = True

    def sample(self, text: Union[str, List[str]], **kwargs) -> Union[List[str], List[List[str]]]:
        """Stub for sampling from the step model (most basic function, just takes the raw input)"""
        raise NotImplementedError('Please implement the sample method for StepModel.')

    @staticmethod
    def format(
            tree: Tree,
            inputs: List[str]
    ) -> str:
        """
        Helper function to take a step and convert it into the raw text input that a step model requires.
        """

        text = [
            tree.get_step_value(x) for x in inputs
        ]

        return " ".join(text)
