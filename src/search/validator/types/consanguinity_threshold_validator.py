from typing import List, Tuple, Union
import itertools

from search.validator import Validator
from search.tree import Tree, StepGeneration
from search.step_type import StepType
from utils.search_utils import normalize


class ConsanguinityThresholdValidator(Validator):

    def __init__(
            self,
            *args,
            threshold: int = None,
            **kwargs
    ):
        """
        Enforces that the inputs used to create a step are unique up to some level of depth.  I.E.

        p0 + p1 -> i0
        p0 + p1 -> i1

        i0 + i1 -> i2 (this step has a consanguinity of 2 because the inputs, i0 and i1, are made of the same ancestors)

        p0 + p0 -> i1 (this step has a consanguinity of 1 because the inputs are exactly the same for the current step.)

        Usually you'll want to set the threshold to be 1 or 2, much higher than that will cause the algorithm to be very
        picky.

        :param args:
        :param threshold: How far up the "family" tree before ancestors can mix.  (1 means siblings cannot match, 2
            means parents cannot overlap, 3 means grandparents, etc. etc.)
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.threshold = threshold


    @staticmethod
    def get_all_ancestor_inputs(tree: Tree, stepname: str, depth: int, current_depth: int = 0):
        """
        Returns a list (including duplicates) of ancestors of some step.

        :param tree: The tree where these steps live
        :param stepname: The name of the step we want to find ancestors for
        :param depth: The depth of which we want to go into the tree from the step
        :param current_depth: The current depth of the search for ancestors
        :return: List of all the steps/ancestors required to make the current step
        """

        # Either we've met the depth specified or we want to ignore the goal node.
        # We ignore the goal node for abductive steps which will often generate from the goal node at some level and all
        # must come from the goal node (any value of the threshold would filter out abductive steps).
        if current_depth >= depth or stepname == 'g':
            return []

        step = tree.get_step(stepname)

        if isinstance(step, str):
            return [stepname]

        return [stepname, *list(itertools.chain(*[
            ConsanguinityThresholdValidator.get_all_ancestor_inputs(tree, x, depth, current_depth=current_depth+1)
            for x in step.inputs
        ]))]

    @staticmethod
    def check_step(tree: Tree, step: StepGeneration, threshold: int):
        """
        Given a step, check to see if it's ancestors overlap within some threshold (distance)

        :param tree: The Tree where all the inputs live
        :param step: The current step we want to validate
        :param threshold: The depth of ancestors to check against
        :return: Either the step or None (none means the step was invalid)
        """

        if len(step.inputs) < 2:
            return step

        ancestors = [
            set([
                ancestor for ancestor in
                ConsanguinityThresholdValidator.get_all_ancestor_inputs(tree, x, depth=threshold) if len(ancestor) > 0
            ])
            for x in step.inputs
        ]

        for idx, ancestor in enumerate(ancestors[0:-1]):
            if sum([len(ancestor.intersection(x)) for x in ancestors[idx + 1:]]) > 0:
                return None

        return step

    def validate(
            self,
            tree: Tree,
            step_type: StepType,
            new_premises: List[str] = (),
            new_hypotheses: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> Tuple[List[str], List[StepGeneration], List[StepGeneration]]:
        """
        Given new hypotheses and intermediates, make sure the inputs used to create those steps have independent sets of
        ancestors up to some depth.

        :param tree: The tree where all the steps live
        :param step_type: The current type of step used to make the generations (ignored here)
        :param new_premises: New premises (ignored here)
        :param new_hypotheses: New hypotheses we want to validate
        :param new_intermediates: New intermediates we want to validate
        :return: New Premises, newly validated hypotheses, and newly validated intermediates as a tuple.
        """

        # Premises have no inputs, so they are always valid
        validated_new_premises = [x for x in new_premises if normalize(x) not in tree.normalized_premises]
        validated_new_hypotheses = []
        validated_new_intermediates = []

        for hypothesis in new_hypotheses:
            hypothesis = self.check_step(tree, hypothesis, self.threshold)
            if hypothesis:
                validated_new_hypotheses.append(hypothesis)

        for intermediate in new_intermediates:
            intermediate = self.check_step(tree, intermediate, self.threshold)
            if intermediate:
                validated_new_intermediates.append(intermediate)

        return validated_new_premises, validated_new_hypotheses, validated_new_intermediates
