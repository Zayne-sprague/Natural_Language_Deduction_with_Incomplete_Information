from typing import Tuple
import torch

from search.entailment.entailment_model import EntailmentModel
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from search.termination.termination_criteria import TerminationCriteria
from search.fringe.fringe_item import Step


class HypothesisAndIntermediateEntailment(TerminationCriteria):

    model: EntailmentModel
    threshold: float

    def __init__(
            self,
            entailment_model_name: str = "wanli_entailment_model",
            entailment_threshold: float = 0.81,
            torch_device: torch.device = torch.device('cpu'),
    ):
        super().__init__()

        # TODO - Pass the model name in somehow (most of this is done via CLI, so another param?)
        self.model = EntailmentModel(model_name=entailment_model_name, torch_device=torch_device)
        self.entailment_threshold = entailment_threshold

    def should_terminate(
            self,
            generation: str,
            tree: Tree,
            step: Step,
            *args,
            specific_type: str = None,
            **kwargs
    ) -> bool:
        """
        If a new hypothesis or a new intermediate have been generated in the last step, check to see if that new step
        generation entails any step of the opposite fringe (i.e. does a new intermediate entail an old hypothesis)

        If a new step gen is entailed, then the tree will collapse the hypotheses into intermediates (deleting all
        the other hypothesis) the only remaining intermediates will be the ones that go from premises directly to
        the goal.
        """

        if len(tree.hypotheses) == 0 or len(tree.intermediates) == 0:
            # This termination criteria is a bridge between an intermediate and a hypothesis, if the tree currently has
            # one or the other but not both, this criteria will never be true.
            return False

        # Figure out if the last step generation was a new intermediate, hypothesis, or neither
        last_hyp = tree.hypotheses[-1]
        last_int = tree.intermediates[-1]

        if (last_hyp.output == generation and specific_type is None) or specific_type == "hypothesis":
            return self.__check_new_hypothesis_entailment__(tree, last_hyp)
        if (last_int.output == generation and specific_type is None) or specific_type == "intermediate":
            return self.__check_new_intermediate_entailment__(tree, last_int)
        return False

    def __check_new_hypothesis_entailment__(
            self,
            tree: Tree,
            hypothesis: StepGeneration
    ) -> bool:
        """Check to see if the new hypothesis is entailed by existing intermediates"""
        for idx, intermediate in enumerate(tree.intermediates):
            if self.model.score(hypothesis.output, intermediate.output) > self.entailment_threshold:
                tree.bridge_intermediate_to_hypotheses(idx, len(tree.hypotheses) - 1)
                return True
        return False

    def __check_new_intermediate_entailment__(
            self,
            tree: Tree,
            intermediate: StepGeneration
    ):
        """Check to see if the new intermediate entails any of the existing hypotheses"""
        for idx, hypothesis in enumerate(tree.hypotheses):
            if self.model.score(hypothesis.output, intermediate.output) > self.entailment_threshold:
                tree.bridge_intermediate_to_hypotheses(len(tree.intermediates) - 1, idx)
                return True
        return False
