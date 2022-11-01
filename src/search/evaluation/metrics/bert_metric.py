from search.evaluation import EvaluationMetric

from bert_score import score
from typing import List


class BertEvaluation(EvaluationMetric):

    def score(self, targets: List[str], predictions: List[str]):
        f1_scores = []

        for target, prediction in zip(targets, predictions):
            _, _, f1 = score([target], prediction, lang='en')
            f1 = f1.item()

            f1_scores.append(f1)

        return f1_scores
