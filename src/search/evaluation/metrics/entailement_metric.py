import torch
from typing import List

from search.evaluation.metrics.evaluation_metric import EvaluationMetric
from search.entailment.entailment_model import EntailmentModel


class EntailmentEvaluation(EvaluationMetric):

    def __init__(self, model_name, torch_device: torch.device = torch.device('cpu')):
        self.model = EntailmentModel(model_name=model_name, torch_device=torch_device)

    def score(self, targets: List[str], predictions: List[str]) -> List[float]:
        return self.model.score(targets=targets, predictions=predictions)
