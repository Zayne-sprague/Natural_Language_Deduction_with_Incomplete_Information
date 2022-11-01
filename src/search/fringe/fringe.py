import heapq
from typing import List, TYPE_CHECKING

from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from search.heuristics.heuristic import Heuristic
from search.fringe.fringe_item import FringeItem
from utils.search_utils import normalize

if TYPE_CHECKING:
    from search.step_type.step_type import StepType


class Fringe:

    tree: Tree
    heuristic: Heuristic

    abductive_queue: List[FringeItem]
    queue: List[FringeItem]
    last_step: str

    def __init__(
            self,
            tree: Tree,
            heuristic: Heuristic,
    ):
        self.tree = tree
        self.heuristic = heuristic

        self.queue = []
        self.abductive_queue = []
        self.last_step = 'neither'

    def populate(
            self,
            step_types: List['StepType'],
            new_premises: List[str] = (),
            new_hypotheses: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ):

        # For each step type, generate the steps that each type could take (unranked)
        all_new_steps = []

        for step_type in step_types:
            all_new_steps.extend(step_type.generate_steps(
                self.tree,
                new_premises,
                new_hypotheses,
                new_intermediates
            ))

        # Update the tree with the new data (make sure the original premises/intermediates/etc. are first to retain
        # ordering (i.e. p1 should still map to the premise in index 1 after we add the new premises.)
        self.tree.premises = [*self.tree.premises, *new_premises]
        self.tree.intermediates = [*self.tree.intermediates, *new_intermediates]
        self.tree.hypotheses = [*self.tree.hypotheses, *new_hypotheses]

        # Score each new step according to the heuristic given
        all_new_scores = self.heuristic.score_steps(self.tree, all_new_steps, step_idx=0)

        # Add all the new steps to the Fringe sorted by the negated heuristic score.
        for step, score in zip(all_new_steps, all_new_scores):
            self.push(FringeItem(-score, step))

    def pop(self) -> FringeItem:
        forward_item = heapq.heappop(self.queue) if len(self.queue) > 0 else None
        abductive_item = heapq.heappop(self.abductive_queue) if len(self.abductive_queue) else None

        next_step = None
        step_type = None

        if self.last_step == 'neither' and abductive_item and forward_item:
            if forward_item:
                step_type = 'forward'
                next_step = forward_item
            else:
                step_type = 'abductive'
                next_step = abductive_item

        elif self.last_step == 'forward' and abductive_item:
            step_type = 'abductive'
            next_step = abductive_item

        elif self.last_step == 'abductive' and forward_item:
            step_type = 'forward'
            next_step = forward_item

        elif forward_item:
            step_type = 'forward'
            next_step = forward_item

        elif abductive_item:
            step_type = 'abductive'
            next_step = abductive_item

        else:
            raise Exception("Could not find anything on the fringe.")

        if step_type == 'forward' and abductive_item:
            self.push(abductive_item)
        if step_type == 'abductive' and forward_item:
            self.push(forward_item)

        self.last_step = step_type
        return next_step

    def push(self, item: FringeItem):
        step_type = item.step.type.name

        if step_type == 'forward':
            heapq.heappush(self.queue, item)
        elif step_type == 'abductive':
            heapq.heappush(self.abductive_queue, item)

    def __len__(self) -> int:
        return len(self.queue) + len(self.abductive_queue)

    def __iter__(self):
        return self

    def __next__(self) -> FringeItem:
        """Helper for iterating through the fringes' priority fringe"""
        try:
            result = self.pop()
            assert result is not None
        except Exception:
            raise StopIteration

        return result
