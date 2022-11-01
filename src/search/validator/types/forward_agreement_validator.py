from typing import List, Tuple
import torch

from search.validator import Validator
from search.tree import Tree, StepGeneration
from search.step_type import StepType, ForwardStepModel
from utils.search_utils import normalize
from search.evaluation.metrics.entailement_metric import EntailmentEvaluation


class ForwardAgreementValidator(Validator):

    def __init__(
            self,
            *args,
            forward_step_model_name: str = None,
            entailment_model_name: str = None,
            torch_device: str = None,
            agreement_threshold: float = None,
            mutual_entailment: bool = False,
            batch_size: int = 4,
            **kwargs
    ):
        """
        Given new hypotheses, this model will attempt to recover the last input from the first input + output of the
        step using the forward model.  If the output is not entailed with a sufficient threshold, the step will be
        dropped. In other words, if a hypothesis is created via

        G - P0 = H1

        We will validate the generated hypothesis (H1) using a forward model via

        P0 + H1 = ~G

        Then we will validate that G == ~G with an entailment model

        :param args:
        :param abductive_step_model_name: The forward model to use defined in PROJECT_ROOT/trained_models/{name}
        :param entailment_model_name: The entailment model to use defined in PROJECT_ROOT/trained_models/{name}
        :param torch_device: THe device to load up the devices on
        :param agreement_threshold: The threshold on the entailment score
        :param mutual_entailment: Should we check mutual entailment on the generation (i.e., G -> ~G, ~G -> G)
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.forward_model = ForwardStepModel(forward_step_model_name, device=torch.device(torch_device), batch_size=batch_size)
        self.entailment_model = EntailmentEvaluation(entailment_model_name, torch_device=torch.device(torch_device))

        self.threshold = agreement_threshold
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
        Given a list of new hypotheses, using a forward model and an entailment model, this function validates the new
        hypotheses by trying to recover the last input of the hypothesis.

        :param tree: The tree where all the steps live
        :param step_type: The step type that was used to generate the hypotheses (ignored here)
        :param new_premises: New premises (ignored here)
        :param new_hypotheses: New Hypotheses we want to validate
        :param new_intermediates: New intermediates (ignored here)
        :return: New Premises, newly validated hypotheses, and new intermediates as a tuple.
        """

        # Premises have no inputs, so they are always valid
        validated_new_premises = new_premises
        validated_new_intermediates = new_intermediates
        validated_new_hypotheses = []

        generation_inputs: List[str] = [
            " ".join([tree.get_step_value(hypothesis.inputs[0]), hypothesis.output])
            for hypothesis in new_hypotheses
        ]
        goals = [
            tree.get_step_value(x.inputs[1]) for x in new_hypotheses
        ]

        forward_generations = self.forward_model.sample(text=generation_inputs, sample_override=1)

        entailment_inputs = []
        entailment_goals = []

        for generations, goal in zip(forward_generations, goals):
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

        for hidx, (hypothesis, generations) in enumerate(zip(new_hypotheses, forward_generations)):

            scores = entailment_scores[hidx]
            if any([x >= self.threshold for x in scores]):
                validated_new_hypotheses.append(hypothesis)

        return validated_new_premises, validated_new_hypotheses, validated_new_intermediates

