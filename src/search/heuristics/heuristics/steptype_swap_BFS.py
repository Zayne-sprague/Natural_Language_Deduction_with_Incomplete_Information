from typing import List, Tuple, Dict, TYPE_CHECKING

from search.tree.tree import Tree
from search.heuristics.heuristic import Heuristic

if TYPE_CHECKING:
    from search.fringe.fringe_item import Step


class StepTypeSwapBFS(Heuristic):
    """
    Same as the BFS heuristic except it will explore different step types before moving onto the next same step that is
    the same type.
    """

    type_counts: Dict[any, int]

    def __init__(self):
        super().__init__()
        self.type_counts = {}

    def score_steps(
            self,
            tree: Tree,
            steps: List['Step'],
            *args,
            **kwargs
    ) -> List[float]:

        scores = []
        for step in steps:
            type_count = self.type_counts.get(step.type.name, 0)
            self.type_counts[step.type.name] = type_count + 1
            scores.append(-type_count)

        return scores
