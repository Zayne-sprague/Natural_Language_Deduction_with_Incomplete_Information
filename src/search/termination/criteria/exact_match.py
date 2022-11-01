from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from search.termination.termination_criteria import TerminationCriteria
from search.fringe.fringe_item import Step

from typing import List


class ExactMatchTermination(TerminationCriteria):

    def should_terminate(
            self,
            generations: List[str],
            tree: Tree,
            step: Step,
            *args,
            **kwargs
    ) -> bool:
        """Exact match termination, if an output of a step is exactly the goal, then stop."""
        # Figure out if the last step generation was a new hypothesis, if so we have to convert it to intermediates.
        last_hyp = tree.hypotheses[-1] if len(tree.hypotheses) > 0 else None

        # If the generation was an intermediate, we just need to trim the intermediates to the subtree that matches
        # the goal only
        last_int = tree.intermediates[-1] if len(tree.intermediates) > 0 else None

        root_intermediate_idx = None

        if last_int and last_int.output == tree.goal and last_int.output == generation:
            root_intermediate_idx = len(tree.intermediates) - 1

        # If we found a root step that matches the goal, let's get its subtree.
        if root_intermediate_idx:
            # Get the subtree of intermediate steps from the step that matched the goal
            subtree = tree.get_subtree(root_intermediate_idx)

            # Set all the intermediates of the tree to only those that lead to directly matching the goal
            tree.intermediates = subtree

            # Remove the hypotheses.
            tree.hypotheses = []
            return True
        return False

        # if generation == tree.goal:
        #     return True
