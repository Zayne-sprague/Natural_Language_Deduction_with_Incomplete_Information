from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from search.step_type.step_type import StepType


@dataclass
class Step:
    inputs: List[str]
    type: 'StepType'


@dataclass(order=True)
class FringeItem:
    """FringeItem that is stored in the queue of the Fringe class"""
    score: float
    step: Step = field(compare=False)
