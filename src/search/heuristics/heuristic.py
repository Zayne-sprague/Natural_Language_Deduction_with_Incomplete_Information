from typing import List, TYPE_CHECKING

from search.tree.tree import Tree
if TYPE_CHECKING:
    from search.fringe.fringe_item import Step


class Heuristic:
    """Class used for scoring steps according to some heuristic"""

    def score_steps(
            self,
            tree: Tree,
            steps: List['Step'],
            *args,
            **kwargs
    ) -> List[float]:

        """Stub for score_steps which returns a list of scores per step."""
        raise NotImplementedError("Implement score_steps for ScoringHeuristic classes")
