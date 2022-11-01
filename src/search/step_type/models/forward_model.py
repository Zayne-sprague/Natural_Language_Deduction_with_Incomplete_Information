from typing import List, Union
import torch

from search.step_type.step_model import StepModel
from step_model.inference import generate_output


class ForwardStepModel(StepModel):
    """The abductive step model used for generating abductions given a step input."""

    def __new__(
            cls,
            model_name: str,
            max_output_length: int = 64,
            num_return_sequences: int = 1,
            batch_size: int = 4,
            device: torch.device = torch.device('cpu'),
            force_new_instance: bool = False
    ):
        """ creates a singleton object, if it is not created,
        or else returns the previous singleton object"""

        instance_name = f'instance__{model_name}_{max_output_length}_{batch_size}_{device}'
        if force_new_instance:
            return super(ForwardStepModel, cls).__new__(cls)

        if not hasattr(cls, instance_name):
            setattr(cls, instance_name, super(ForwardStepModel, cls).__new__(cls))
        return getattr(cls, instance_name)

    def sample(self, text: Union[str, List[str]], sample_override=None, **kwargs) -> Union[List[str], List[List[str]]]:
        samples = sample_override if sample_override is not None else self.num_return_sequences

        _, generated, _ = generate_output(
            text,
            self.tokenizer,
            self.model,
            self.device,
            batch_size=self.batch_size,
            num_return_sequences=samples
        )

        generated = [output.strip('<pad> ').strip('</s>') for output in generated]

        if isinstance(text, str):
            return generated
        else:
            batches = []
            for gen_idx in range(len(text)):
                batch = []
                for sample_idx in range(samples):
                    batch.append(generated[(gen_idx * samples) + sample_idx])
                batches.append(batch)
            return batches
