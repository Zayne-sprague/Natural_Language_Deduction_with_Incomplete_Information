from typing import List


class EvaluationMetric:

    def score(self, targets: List[str], predictions: List[str]) -> List[float]:
        raise NotImplemented('Implement this per evaluation child class.')