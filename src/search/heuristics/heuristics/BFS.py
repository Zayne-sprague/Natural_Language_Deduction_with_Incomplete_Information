from typing import List, Tuple

from search.tree.tree import Tree
from search.heuristics.heuristic import Heuristic


class BreadthFirstSearchHeuristic(Heuristic):
    """
    Rank all steps the same value which will mean the algorithm takes each step in order.
    """

    step_idx: int

    def __init__(self):
        super().__init__()
        self.step_idx = 0

    def score_steps(
            self,
            tree: Tree,
            steps: List[Tuple[str, str]],
            *args,
            **kwargs
    ) -> List[float]:
        self.step_idx = self.step_idx + 1
        return [-self.step_idx]*len(steps)
