from search.evaluation.metrics.evaluation_metric import EvaluationMetric
from search.evaluation.metrics.rouge_metric import RougeEvaluation
from search.evaluation.metrics.entailement_metric import EntailmentEvaluation

from typing import List


class RougeEntailmentHMMetric(EvaluationMetric):

    rouge_scorer: RougeEvaluation
    entailment_scorer: EntailmentEvaluation

    def __init__(self, rouge_scorer: RougeEvaluation, entailment_scorer: EntailmentEvaluation):
        self.rouge_scorer = rouge_scorer
        self.entailment_scorer = entailment_scorer

    def score(self, targets: List[str], predictions: List[str]) -> List[float]:
        e_score = self.entailment_scorer.score(targets, predictions)
        r_score = self.rouge_scorer.score(targets, predictions)

        return [(2 * e * r) / (e + r) for e, r in zip(e_score, r_score)]
