from typing import List, Dict

from search.evaluation.metrics.evaluation_metric import EvaluationMetric


class Evaluation:
    """
    Given a target sentence and a predicted sentence, hold scores measuring their likeness.
    """

    registered_evaluation_metrics: List[EvaluationMetric]

    def __init__(
            self,
    ):
        self.registered_evaluation_metrics = []

    def register_evaluation_metric(self, metric: EvaluationMetric):
        self.registered_evaluation_metrics.append(metric)

    def remove_evaluation_metric(self, metric: EvaluationMetric):
        self.registered_evaluation_metrics.remove(metric)

    @property
    def metrics(self) -> Dict[str, any]:
        raise NotImplementedError("Implement the metrics property of the Evaluation class")

    def to_json(self) -> Dict[str, any]:
        raise NotImplementedError("Implement to_json for Evaluation classes")

    @classmethod
    def from_json(cls, data: Dict[str, any]) -> 'Evaluation':
        raise NotImplementedError("Implement from_json for Evaluation classes")
