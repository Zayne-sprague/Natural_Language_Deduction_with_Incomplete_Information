from typing import List, Tuple, ClassVar, Iterable, Dict
from dataclasses import dataclass, field

import torch
import transformers
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from step_model.inference import get_model, get_tokenizer, generate_output
from utils.paths import TRAINED_MODELS_FOLDER
from utils.search_utils import normalize
from search.tree import Tree, StepGeneration
from search.step_type import StepType
from search.fringe.fringe import Fringe
from search.termination.termination_criteria import TerminationCriteria
from search.termination.criteria.exact_match import ExactMatchTermination
from search.heuristics.heuristic import Heuristic
from search.heuristics.heuristics.BFS import BreadthFirstSearchHeuristic
from search.validator import Validator


class Search:
    termination_criteria: List[TerminationCriteria]
    max_steps: int

    def __init__(
            self,
            step_types: List[StepType] = (),
            termination_criteria: List[TerminationCriteria] = (ExactMatchTermination()),
            max_steps: int = 6
    ):
        """
        Base search class that will handle the over-arching search and fringe.  It can use multiple step type models
        and prioritize their individual steps executing one step at a time.

        :param step_types: What type of steps are valid to take in the Search (abductive, forward, etc.)
        :param termination_criteria: What types of termination criteria can end the search early (TODO - move this to
            the function search itself?)
        :param max_steps: Number of max steps to take in a search (TODO - move this to teh function search itself?)
        """

        self.step_types = step_types

        self.termination_criteria = termination_criteria
        self.max_steps = max_steps

    def search(
            self,
            goal: str,
            premises: List[str],
            hypotheses: List[StepGeneration],
            heuristic: ClassVar[Heuristic] = BreadthFirstSearchHeuristic,
            termination_criteria: List[TerminationCriteria] = (),
            validators: List[Validator] = (),
            max_steps: int = None,
            show_progress: bool = False,
            pbar_kwargs: Dict[str, any] = None,
    ) -> Tree:
        """
        The actual search.

        Given a list of premises, a hypothesis, and a goal state you want to reach (could be the hypothesis a second
        time) -- iterate through the step types to create new prioritized steps and execute them one at a time until
        either a termination criterion is met or we hit the maximum number of steps.

        The heuristic will rank the individual steps.

        :param goal: The goal the proof tree will be oriented towards solving
        :param premises: The list of premises to start the search with
        :param hypotheses: A list of hypotheses to begin the search with (can be blank)
        :param heuristic: The heuristic used to guide the search and which step to take from the fringe.
        :param termination_criteria: What termination criteria can end the search
        :param validators: What validators should be used to determine if a steps generation is correct.
        :param max_steps: The maximum number of steps to perform in the search
        :param show_progress: Show a progress bar
        :param pbar_kwargs: Key word arguments for the progress bar
        :return: An expanded Tree object with intermediates and hypotheses created during the inner loop of the search
            (does not always mean the search found the goal)
        """

        termination_criteria = self.termination_criteria if not termination_criteria else termination_criteria
        max_steps = self.max_steps if not max_steps else max_steps
        pbar_kwargs = pbar_kwargs if pbar_kwargs else {'desc': 'Searching The Fringe', 'total': max_steps}

        # Build the fringe with the initial premises and hypothesis.
        fringe = Fringe(
            Tree(goal=goal),
            heuristic=heuristic()
        )

        fringe.populate(self.step_types, premises, hypotheses)

        iterator = tqdm(enumerate(fringe), **pbar_kwargs) if show_progress else enumerate(fringe)


        # Inner loop of the search, sample the top most item from the fringes priority queue.
        for step_idx, item in iterator:
            # Get the data for the step (step model, step inputs, step type... step everything!)
            step = item.step
            inputs = step.inputs
            step_type: StepType = step.type
            step_model = step_type.step_model

            # Get step generations from the step model.
            formatted_input = step_model.format(fringe.tree, inputs)
            step_generations = step_model.sample(formatted_input)

            # Each step type generates a different output (forward generates intermediates, abductive generate
            # hypotheses for example) - instead of a big if statement, we allow the class to return the new items.
            new_premises, new_hypotheses, new_intermediates = step_type.generation_to_step_generation(
                step_generations,
                step
            )

            # Check with each validator and make sure the new generated statements are valid.
            for validator in validators:
                new_premises, new_hypotheses, new_intermediates = validator.validate(
                    tree=fringe.tree,
                    step_type=step_type,
                    new_premises=new_premises,
                    new_hypotheses=new_hypotheses,
                    new_intermediates=new_intermediates
                )

            # Only populate the fringe if something new was generated (otherwise skip it).
            if len(new_premises) > 0 or len(new_intermediates) > 0 or len(new_hypotheses) > 0:
                # Given the new generation and its type (premise, hypothesis, or intermediate) update the fringe
                # with the new output and score it so it gets prioritized.
                fringe.populate(self.step_types, new_premises, new_hypotheses, new_intermediates)

            # # Did the step generation meet any of the criteria for termination?
            # TODO - refactor this to account for the type of step rather than just generations
            #    (i.e. pass in the new premises/hypotheses/intermediates not the generations themselves)
            # should_terminate = any([
            #     x.should_terminate(step_generations, fringe.tree, step) for x in termination_criteria
            # ])

            # if should_terminate:
            #     return fringe.tree

            # Max step exit
            if (step_idx + 1) >= max_steps:
                break

        # Return the fringes tree that contains all the added generations.
        return fringe.tree