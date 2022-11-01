from typing import List, Tuple

from search.step_type.step_type import StepType
from search.tree.tree import Tree
from search.fringe.fringe_item import Step
from search.tree.step_generation import StepGeneration


class BackwardStepType(StepType):

    name: str = 'backward'

    def generate_steps(
            self,
            tree: Tree,
            new_premises: List[str] = (),
            new_hypothesis: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> List[Step]:
        raise Exception("TODO - Implement Backward Step Type")

    def generation_to_step_generation(
            self,
            generation: str, step: 'Step'
    ) -> Tuple[List[str], List[StepGeneration], List[StepGeneration]]:
        raise Exception("TODO - Implement Backward Step Type")
