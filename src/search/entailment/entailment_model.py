from utils.paths import TRAINED_MODELS_FOLDER
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset
from typing import Union, List


class EntailmentModel:

    def __new__(
            cls,
            model_name: str,
            torch_device: torch.device = torch.device('cpu'),
            batch_size: int = 4,
            force_new_instance: bool = False
    ):
        """ creates a singleton object, if it is not created,
        or else returns the previous singleton object"""

        instance_name = f'instance__{model_name}_{batch_size}_{torch_device}'
        if force_new_instance:
            return super(EntailmentModel, cls).__new__(cls)

        if not hasattr(cls, instance_name):
            setattr(cls, instance_name, super(EntailmentModel, cls).__new__(cls))
        return getattr(cls, instance_name)

    def __init__(
            self,
            model_name: str,
            torch_device: torch.device = torch.device('cpu'),
            batch_size: int = 4,
            force_new_instance: bool = False
    ):
        if hasattr(self, 'instantiated'):
            return

        self.model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODELS_FOLDER / model_name)

        if 'cuda' in torch_device.type and torch.cuda.is_available():
            self.model.to(torch_device)
        else:
            self.model.to('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODELS_FOLDER / model_name)

        self.batch_size = batch_size

        self.instantiated = True

    def score(self, targets: Union[List[str], str], predictions: Union[List[str], str]):
        """

        :param targets: List of target sentences you want to see if is entailed by a prediction
        :param predictions: A list of predictions you want to compare against the target
        :return: The ENTAILMENT probability per prediction target combo
        """

        if isinstance(predictions, str):
            predictions = [predictions] * (1 if isinstance(targets, str) else len(targets))
        if isinstance(targets, str):
            targets = [target] * len(predictions)

        dataset = Dataset.from_dict({"inputs": predictions, "targets": targets})
        dataset = dataset.map(lambda e: e, batched=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        all_probabilities = []
        for batch in dataloader:
            inputs_encoded = self.tokenizer(
                batch['inputs'],
                text_pair=batch['targets'],
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            inputs_encoded.data = {
                k: v.to(self.model.device)
                if isinstance(v, torch.Tensor) else
                v
                for k, v in inputs_encoded.data.items()
            }

            # This model will produce 3 logits for 3 classes ENTAIL, NEUTRAL, and CONTRADICT
            # We are only interested in the entailment class probability usually which is why we only look at the last
            # index
            logits = self.model(**inputs_encoded).logits.detach().cpu()
            probs = torch.nn.functional.softmax(logits, dim=1)[:, -1].tolist()
            all_probabilities.extend(probs)

        return all_probabilities

