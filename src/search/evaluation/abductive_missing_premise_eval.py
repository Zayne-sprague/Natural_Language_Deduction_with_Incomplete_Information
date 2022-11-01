from typing import Dict, Tuple, List

from search.evaluation.evaluation import Evaluation
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration


class AbductiveMissingPremiseEvaluation(Evaluation):

    predicted_tree: Tree
    gold_tree: Tree

    def __init__(
            self,
            predicted_tree: Tree,
            gold_tree: Tree,
    ):
        super().__init__()

        self.predicted_tree = predicted_tree
        self.gold_tree = gold_tree

    @property
    def metrics(self) -> Dict[str, any]:
        return {}

    @property
    def exact_premise_matches(self) -> Tuple[int, int]:
        gold_premises = set(self.gold_tree.premises)
        predicted_premises = set(self.predicted_tree.premises)

        missing_premises = gold_premises - predicted_premises

        target_intermediates = self.predicted_tree.intermediates
        outputs = [x.output for x in target_intermediates]

        exact_matches = sum([1 if x in missing_premises else 0 for x in outputs])
        return exact_matches, len(gold_premises)

    @property
    def targets(self) -> List[str]:
        gold_premises = set(self.gold_tree.premises)
        target_premises = set(self.predicted_tree.premises)

        missing_premises = gold_premises - target_premises
        return list(missing_premises)


    def to_json(self) -> Dict[str, any]:
        return {
            'gold_tree': self.gold_tree.to_json(),
            'predicted_tree': self.predicted_tree.to_json()
        }

    @classmethod
    def from_json(cls, data) -> 'AbductiveMissingPremiseEvaluation':
        gold_tree = Tree.from_json(data['gold_tree'])
        predicted_tree = Tree.from_json(data['predicted_tree'])

        return AbductiveMissingPremiseEvaluation(gold_tree=gold_tree, predicted_tree=predicted_tree)

