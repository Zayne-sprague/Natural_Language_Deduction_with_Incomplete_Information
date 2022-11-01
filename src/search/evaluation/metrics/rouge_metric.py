from search.evaluation.metrics.evaluation_metric import EvaluationMetric
from rouge_score import rouge_scorer
from typing import List


class RougeEvaluation(EvaluationMetric):

    rouge_type: str
    use_stemmer: bool
    scorer: rouge_scorer.RougeScorer

    def __init__(self, rouge_type: str = 'rouge1', use_stemmer: bool = True):
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer

        self.scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)

    def score(self, targets: [str], predictions: List[str]) -> List[float]:

        fmeasures = []

        for target, prediction in zip(targets, predictions):
            scores = self.scorer.score(target, prediction)
            measures = scores[self.rouge_type]
            fmeasures.append(measures.fmeasure)

        return fmeasures
