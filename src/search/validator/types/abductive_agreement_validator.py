from typing import List, Tuple
import torch
from copy import deepcopy

from search.validator import Validator
from search.tree import Tree, StepGeneration
from search.step_type import StepType, AbductiveStepModel
from utils.search_utils import normalize
from search.evaluation.metrics.entailement_metric import EntailmentEvaluation


class AbductiveAgreementValidator(Validator):
    """
    Given new intermediate, this model will attempt to recover each input given the output of the intermediate step and
    the other input.  If the output is not entailed with a sufficient threshold, the step will be dropped. This is
    repeated for all inputs.
    """

    def __init__(
            self,
            *args,
            abductive_step_model_name: str = None,
            entailment_model_name: str = None,
            torch_device: str = None,
            agreement_threshold: float = None,
            invalid_input_tolerance: int = 0,
            mutual_entailment: bool = False,
            **kwargs
    ):
        """
        This class validates new intermediate generations by recovering the inputs that were used to make the new
        intermediate via an abduction model and an entailment model.

        :param args:
        :param abductive_step_model_name: The abductive model to use defined in PROJECT_ROOT/trained_models/{name}
        :param entailment_model_name: The entailment model to use defined in PROJECT_ROOT/trained_models/{name}
        :param torch_device: The device to load the models onto
        :param agreement_threshold: The entailment threshold required for validating a generation vs an input
        :param invalid_input_tolerance: Do we allow N inputs to not be recovered (0 means every premise must be
            recovered via the abductive model + entailment model)
        :param mutual_entailment: Do we check entailment in both directions (generation -> input, input -> generation)
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.abductive_model = AbductiveStepModel(abductive_step_model_name, device=torch.device(torch_device))
        self.entailment_model = EntailmentEvaluation(entailment_model_name, torch_device=torch.device(torch_device))

        self.threshold = agreement_threshold
        self.invalid_input_tolerance = invalid_input_tolerance
        self.mutual_entailment = mutual_entailment

    def validate(
            self,
            tree: Tree,
            step_type: StepType,
            new_premises: List[str] = (),
            new_hypotheses: List[StepGeneration] = (),
            new_intermediates: List[StepGeneration] = ()
    ) -> Tuple[List[str], List[StepGeneration], List[StepGeneration]]:
        """
        This function is meant to validate new intermediates by trying to recover the individual inputs used to create
        the new intermediate via the abductive model.  In other words, if a new intermediate is of the form

        P1 + P2 = I0

        We validate this by doing

        I0 - P1 = ~P2
        I0 - P2 = ~P1

        We then check if P1 == ~P1 and P2 == ~P2 within some entailment threshold.

        :param tree: The current tree that has the inputs for the intermediate that we can look up
        :param step_type: The step we took to generate the new intermediates
        :param new_premises: New premises (they'll be ignored here)
        :param new_hypotheses: New hypotheses (they'll be ignored here)
        :param new_intermediates: New Intermediates we want to validate
        :return: All the new premises, new hypotheses, and the newly validated intermediates as a tuple.
        """

        # Premises have no inputs, so they are always valid
        validated_new_premises = new_premises
        validated_new_intermediates = []
        validated_new_hypotheses = new_hypotheses

        generation_inputs = [
            " ".join([tree.get_step_value(arg), intermediate.output])
            for intermediate in new_intermediates for arg in intermediate.inputs
        ]

        goals = [
            tree.get_step_value(arg) for x in new_intermediates for arg in x.inputs[::-1]
        ]

        abductive_generations = self.abductive_model.sample(text=generation_inputs, sample_override=1)

        entailment_inputs = []
        entailment_goals = []

        for generations, goal in zip(abductive_generations, goals):
            if isinstance(generations, str):
                entailment_inputs.append(generations)
                entailment_goals.append(goal)
            else:
                entailment_inputs.extend(generations)
                entailment_goals.extend([goal] * len(generations))

        g2p = self.entailment_model.score(entailment_goals, entailment_inputs)
        if self.mutual_entailment:
            p2g = self.entailment_model.score(entailment_inputs, entailment_goals)
            entailment_scores = [(x + y) / 2 for x, y in zip(g2p, p2g)]
        else:
            entailment_scores = g2p

        entailment_scores = [
            entailment_scores[i:i+1]
            for i in range(0, len(entailment_scores), 1)
        ]

        abductive_generations = [
            [abductive_generations[i], abductive_generations[i+1]] for i in range(0, len(abductive_generations), 2)
        ]
        entailment_scores = [
            [entailment_scores[i], entailment_scores[i+1]] for i in range(0, len(entailment_scores), 2)
        ]

        for iidx, (intermediate, args_generations) in enumerate(zip(new_intermediates, abductive_generations)):
            invalid_args = 0

            for aidx, arg_generations in enumerate(args_generations):

                scores = entailment_scores[iidx][aidx]

                if any([x >= self.threshold for x in scores]):
                    break
                invalid_args += 1

                if invalid_args > self.invalid_input_tolerance:
                    break

            if invalid_args <= self.invalid_input_tolerance:
                validated_new_intermediates.append(intermediate)

        return validated_new_premises, validated_new_hypotheses, validated_new_intermediates
