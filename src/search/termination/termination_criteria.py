from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from search.fringe.fringe_item import Step

from typing import List


class TerminationCriteria:
    """Class to help with termination criteria for the search"""

    def should_terminate(
            self,
            generations: List[str],
            tree: Tree,
            step: Step,
            *args,
            **kwargs
    ) -> bool:
        """Stub for the should_terminate function determining if the algorithm should stop."""
        raise NotImplementedError('Implement should_terminate for all termination criteria')
