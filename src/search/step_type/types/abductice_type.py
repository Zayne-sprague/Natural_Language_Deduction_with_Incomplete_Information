from typing import List, Tuple

from search.step_type.step_type import StepType
from search.tree.tree import Tree
from search.fringe.fringe_item import Step, FringeItem
from search.tree.step_generation import StepGeneration
from search.fringe.fringe import Fringe


class AbductiveStepType(StepType):
    """Abductive Model G + C -> I, I + A -> B"""

    name: str = 'abductive'

    def generate_steps(
            self,
            tree: Tree,
            new_premises: List[str] = (),
            new_hypothesis: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> List[Step]:
        new_steps: List[Step] = []
        for nidx, _ in enumerate(new_premises):
            new_premise_index = nidx + len(tree.premises)

            new_steps.append(
                Step([f'p{new_premise_index}', 'g'], self)
            )

            # for iidx, _ in enumerate([*tree.intermediates, *new_intermediates]):
            #     new_steps.append(
            #         Step([f'p{new_premise_index}', f'i{iidx}'], self)
            #     )

            for hidx, _ in enumerate(tree.hypotheses):
                new_steps.append(
                    Step([f'i{new_premise_index}', f'h{hidx}'], self)
                )

        for nidx, _ in enumerate(new_intermediates):
            new_intermediate_index = len(tree.intermediates) + nidx

            # for pidx, _ in enumerate(tree.premises):
            #     new_steps.append(
            #         Step([f'i{new_intermediate_index}', f'p{pidx}'], self)
            #     )

            for hidx, _ in enumerate(tree.hypotheses):
                new_steps.append(
                    Step([f'i{new_intermediate_index}', f'h{hidx}'], self)
                )

        for nidx, _ in enumerate(new_hypothesis):
            new_hypothesis_index = len(tree.hypotheses) + nidx

            for pidx, _ in enumerate(tree.premises):
                new_steps.append(
                    Step([f'p{pidx}', f'h{new_hypothesis_index}'], self)
                )

            for iidx, _ in enumerate([*tree.intermediates, *new_intermediates]):
                new_steps.append(
                    Step([f'i{iidx}', f'h{new_hypothesis_index}'], self)
                )

        filtered_new_steps = []
        allowed = []

        for step in new_steps:
            ins = " ".join([step.inputs[0], step.inputs[1]])
            if ins in allowed:
                continue

            allowed.append(ins)
            filtered_new_steps.append(step)

        return new_steps

    def generation_to_step_generation(
            self,
            generations: List[str],
            step: Step,
    ) -> Tuple[List[str], List[StepGeneration], List[StepGeneration]]:
        hypotheses = []
        for generation in generations:
            hypotheses.append(StepGeneration(step.inputs, generation, tags={'step_type': 'abductive'}))
        return [], hypotheses, []

