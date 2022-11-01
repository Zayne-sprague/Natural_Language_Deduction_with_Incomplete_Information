from typing import List, Tuple

from search.tree.tree import Tree
from search.heuristics.heuristic import Heuristic


class DepthFirstSearchHeuristic(Heuristic):
    """
    Rank all steps the same value which will mean the algorithm takes each step in order.
    """

    def score_steps(
            self,
            tree: Tree,
            steps: List[Tuple[str, str]],
            *args,
            **kwargs
    ) -> List[float]:
        return [0]*len(steps)
