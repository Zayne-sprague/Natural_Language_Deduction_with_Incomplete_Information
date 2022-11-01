from typing import List, Tuple, TYPE_CHECKING

from search.tree.step_generation import StepGeneration
from search.step_type.step_model import StepModel
from search.tree.tree import Tree
from search.fringe.fringe import Fringe

if TYPE_CHECKING:
    from search.fringe.fringe_item import Step


class StepType:

    """
    StepTypes are responsible for holding the step model that will run inference on the step inputs
    as well as generating new steps when given a list of new premises, intermediates or a new hypothesis.

    They also keep track of their respective names so you can look them up.
    """

    step_model: StepModel
    name: str

    def __init__(self, step_model: StepModel):
        self.step_model = step_model

    def generate_steps(
            self,
            tree: Tree,
            new_premises: List[str] = (),
            new_hypothesis: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> List['Step']:
        """
        This function is responsible for creating steps (i.e. 'p1', 'p2') that will be added to the fringes priority
        queue along with the step types name.  All that is required is to return a list of steps.

        :param tree: The current tree that holds all the state of the search (premises, hypotheses, intermediates)
        :param new_premises: A list of new premises that have been created but not added to the fringe yet.
        :param new_hypothesis: A new hypothesis that will be set in the fringe
        :param new_intermediates: A list of new Intermediates that have been created but not added to the fringe yet.
        """
        raise NotImplementedError

    def generation_to_step_generation(
            self,
            generations: List[str],
            step: 'Step',
    ) -> Tuple[List[str], List[StepGeneration], List[StepGeneration]]:
        raise NotImplementedError('Please implement the add_generations_to_fringe for StepModel')
