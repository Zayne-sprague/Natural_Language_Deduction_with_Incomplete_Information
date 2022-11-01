from abc import ABC, abstractmethod
from typing import List, Tuple

from search.tree import Tree, StepGeneration
from search.step_type import StepType


class Validator(ABC):
    """
    This class is responsible for taking a generation from a stepmodel and
    determining if that generated statement is valid.  If it's not valid, the
    generation will be removed and not added to the fringe.

    Multiple validations can exist and not all Validator types have to be
    stepmodel specific (i.e. : "is a generated statement one of the inputs")
    """

    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def validate(
            self,
            tree: Tree,
            step_type: StepType,
            new_premises: List[str] = (),
            new_hypotheses: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> Tuple[List[str], List[StepGeneration], List[StepGeneration]]:
        """
        This is the base class that will be called when new premises, hypotheses, or intermediates are generated.

        :param tree: The tree with all the current search state
        :param step_type: The type of step that generated the outputs
        :param new_premises: New premises that were generated as an output of a step model.
        :param new_hypotheses: New hypotheses that were generated as an output of a step model.
        :param new_intermediates: New intermediates that were generated as an output of a step model.
        :return: A tuple of the filtered -- filtered new premises, filtered new hypotheses, filtered new intermediates
        """
        return new_premises, new_hypotheses, new_intermediates
