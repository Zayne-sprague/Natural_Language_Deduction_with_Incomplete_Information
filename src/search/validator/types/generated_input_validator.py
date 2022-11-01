from typing import List, Tuple

from search.validator import Validator
from search.tree import Tree, StepGeneration
from search.step_type import StepType
from utils.search_utils import normalize


class GeneratedInputValidator(Validator):
    """
    Simple Validator that checks to make sure any new generation does not EXACTLY match the inputs (fuzzy matching is
    still okay).
    """

    def validate(
            self,
            tree: Tree,
            step_type: StepType,
            new_premises: List[str] = (),
            new_hypotheses: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> Tuple[List[str], List[StepGeneration], List[StepGeneration]]:
        """
        Checks whether the generation from a step was a copy of the inputs OR if the newly generated output has already
        been generated in that corresponding step type (no duplicate interemdiates or hypotheses, but an intermediate
        could generate the same output as a hypothesis)

        :param tree: The tree were all the inputs exist
        :param step_type: The type of step used to generate the new data
        :param new_premises: New premises to validate
        :param new_hypotheses: New hypotheses to validate
        :param new_intermediates: New intermediates to validate
        :return: Newly validated premises, newly validated hypotheses, and newly validated intermediates as a tuple
        """

        validated_new_premises = [x for x in new_premises if normalize(x) not in tree.normalized_premises]
        validated_new_hypotheses = []
        validated_new_intermediates = []

        for hypothesis in new_hypotheses:
            inputs = [tree.get_step_value(x) for x in hypothesis.inputs]
            if hypothesis.output in inputs or hypothesis.output in [x.output for x in validated_new_hypotheses]:
                continue
            validated_new_hypotheses.append(hypothesis)

        for intermediate in new_intermediates:
            inputs = [tree.get_step_value(x) for x in intermediate.inputs]
            if intermediate.output in inputs or intermediate.output in [x.output for x in validated_new_intermediates]:
                continue
            validated_new_intermediates.append(intermediate)

        validated_new_hypotheses = [x for x in validated_new_hypotheses if normalize(x.output) not in tree.normalized_hypotheses]
        validated_new_intermediates = [x for x in validated_new_intermediates if normalize(x.output) not in tree.normalized_intermediates]

        return validated_new_premises, validated_new_hypotheses, validated_new_intermediates
