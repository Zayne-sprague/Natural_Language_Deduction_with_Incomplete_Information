from typing import List, Union

from search.step_type.step_model import StepModel


class BackwardStepModel(StepModel):
    """The backwards step model used for generating intermediates or premises given a hypothesis or goal."""

    def sample(self, text: Union[str, List[str]], **kwargs) -> Union[List[str], List[List[str]]]:
        raise Exception("TODO - Implement backward model")
