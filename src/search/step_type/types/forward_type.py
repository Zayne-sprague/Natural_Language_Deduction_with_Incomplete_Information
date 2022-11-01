from typing import List, Tuple

from search.step_type.step_type import StepType
from search.tree.tree import Tree
from search.fringe.fringe_item import Step
from search.tree.step_generation import StepGeneration


class ForwardStepType(StepType):
    """Forward model, A + B -> I, I + C -> G"""

    name: str = 'forward'

    def generate_steps(
            self,
            tree: Tree,
            new_premises: List[str] = (),
            new_hypothesis: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> List[Step]:

        new_steps: List[Step] = []

        for nidx, _ in enumerate(new_premises):
            new_premise_index = nidx + max([len(tree.premises) - 1, 0])

            for pidx, _ in enumerate([*tree.premises, *new_premises]):
                if new_premise_index != pidx:
                    new_p_idx = new_premise_index + len(tree.premises)
                    new_steps.append(
                        Step([f'p{new_p_idx}', f'p{pidx}'], self)
                    )

                    new_steps.append(
                        Step([f'p{pidx}', f'p{new_p_idx}'], self)
                    )

            for iidx, _ in enumerate([*tree.intermediates, *new_intermediates]):
                new_steps.append(
                    Step([f'p{new_premise_index}', f'i{iidx}'], self)
                )

                new_steps.append(
                    Step([f'i{iidx}', f'p{new_premise_index}'], self)
                )

        for nidx, _ in enumerate(new_intermediates):
            new_intermediate_index = len(tree.intermediates) + nidx

            for iidx, _ in enumerate([*tree.intermediates, *new_intermediates]):

                if new_intermediate_index != iidx and len(tree.intermediates) > 0:
                    new_steps.append(
                        Step([f'i{new_intermediate_index}', f'i{iidx}'], self)
                    )

                    new_steps.append(
                        Step([f'i{iidx}', f'i{new_intermediate_index}'], self)
                    )

            for pidx, _ in enumerate([*tree.premises, *new_premises]):

                new_steps.append(
                    Step([f'p{pidx}', f'i{new_intermediate_index}'], self)
                )
                new_steps.append(
                    Step([f'i{new_intermediate_index}', f'p{pidx}'], self)
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
        intermediates = []
        for generation in generations:
            intermediates.append(StepGeneration(step.inputs, generation, tags={'step_type': 'forward'}))
        return [], [], intermediates
