import random

from search.evaluation.abductive_missing_premise_eval import AbductiveMissingPremiseEvaluation
from search.tree.tree import Tree
from search.heuristics.heuristics.BFS import BreadthFirstSearchHeuristic
from search.heuristics.heuristics.DFS import DepthFirstSearchHeuristic
from search.heuristics.heuristics.steptype_modeled import StepTypeModeled
from search.heuristics.heuristics.steptype_swap_BFS import StepTypeSwapBFS
from search.step_type.types.backward_type import BackwardStepType
from search.step_type.types.forward_type import ForwardStepType
from search.step_type.types.abductice_type import AbductiveStepType
from search.step_type.models.forward_model import ForwardStepModel
from search.step_type.models.backward_model import BackwardStepModel
from search.step_type.models.abductive_model import AbductiveStepModel
from search.termination.criteria.exact_match import ExactMatchTermination
from search.termination.criteria.hypothesis_and_intermediate_entailment import \
    HypothesisAndIntermediateEntailment
from search.validator import GeneratedInputValidator, ForwardAgreementValidator, \
    AbductiveAgreementValidator, ConsanguinityThresholdValidator
from search.search import Search
from utils.config_handler import merge_yaml_and_namespace

import json
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional
from functools import partial
import torch
from multiprocessing import Pool

import json


def __process_wrapper__(kwargs):
    return __search__(**kwargs)


def search(
    input_file: Path,
    output_file: Path,
    force_output: bool,
    heuristic: str,
    step_types: List[str],
    step_model_names: List[str],
    nums_return_sequences: List[int],
    termination_criteria: List[str],
    validator_names: List[str],
    forward_agreement_validator_forward_model_name: str,
    forward_agreement_validator_entailment_model_name: str,
    forward_agreement_validator_agreement_threshold: float,
    forward_agreement_validator_mutual_entailment: bool,
    abductive_agreement_validator_abductive_model_name: str,
    abductive_agreement_validator_entailment_model_name: str,
    abductive_agreement_validator_agreement_threshold: float,
    abductive_agreement_validator_invalid_input_tolerance: int,
    abductive_agreement_validator_mutual_entailment: bool,
    consanguinity_threshold_validator_threshold: int,
    max_steps: int,
    torch_devices: List[str],
    all_one_premise: bool = False,
    mix_distractors: bool = False,
    silent: bool = False
):
    """
    Wrapper function for __search__

    Specifically, this wrapper helps spread the search across multiple devices by splitting the number of trees to
    search over.

    This splitting is done via multiprocessing and putting each instance of the search on a new core. (might not work if
    you have more devices than cpu cores... but lucky you if that's the case :D )
    """

    # MAKE SURE ARGUMENTS FOR IO ARE OKAY
    valid_validator_choices = ['GeneratedInput', 'ForwardAgreement', 'ConsanguinityThreshold',
                               'AbductiveAgreement']
    assert input_file.is_file() and str(input_file).endswith('.json'), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'
    assert len(step_types) == len(step_model_names), \
        'Please specify a model name (name of folder in {DEDUCE}/trained_models/*) per each step_type given.'
    assert max_steps > 0, 'Please specify a --max_steps (-ms) value that is > 0.'
    assert all([x in valid_validator_choices for x in validator_names]), \
        f'Please specify a correct validator, choices are: {", ".join(valid_validator_choices)}'
    assert len(step_types) == len(nums_return_sequences) or len(nums_return_sequences) == 1, \
        f'Either specify a num_return_sequences for each step model or choose 1 value to be shared across all models.'

    # LOAD UP TREES
    json_trees = json.load(input_file.open('r'))
    all_trees = [Tree.from_json(t) for t in json_trees]
    device_trees = []

    trees_per_device = len(all_trees) / len(torch_devices)
    for i in range(len(torch_devices)):
        start_idx = int(min(i * trees_per_device, len(all_trees) - 1))
        end_idx = int(min((i+1) * trees_per_device, len(all_trees)))

        if i == len(torch_devices) - 1:
            # If its the last device, always take all the trees up till the last index.
            end_idx = len(all_trees)

        device_trees.append(all_trees[start_idx:end_idx])

    searches = []
    for idx, (torch_device, trees) in enumerate(zip(torch_devices, device_trees)):

        search_args = {
            'trees': trees,
            'heuristic': heuristic,
            'step_types': step_types,
            'step_model_names': step_model_names,
            'nums_return_sequences': nums_return_sequences,
            'termination_criteria': termination_criteria,
            'validator_names': validator_names,
            'forward_agreement_validator_forward_model_name': forward_agreement_validator_forward_model_name,
            'forward_agreement_validator_entailment_model_name': forward_agreement_validator_entailment_model_name,
            'forward_agreement_validator_agreement_threshold': forward_agreement_validator_agreement_threshold,
            'forward_agreement_validator_mutual_entailment': forward_agreement_validator_mutual_entailment,
            'abductive_agreement_validator_abductive_model_name': abductive_agreement_validator_abductive_model_name,
            'abductive_agreement_validator_entailment_model_name': abductive_agreement_validator_entailment_model_name,
            'abductive_agreement_validator_agreement_threshold': abductive_agreement_validator_agreement_threshold,
            'abductive_agreement_validator_invalid_input_tolerance': abductive_agreement_validator_invalid_input_tolerance,
            'abductive_agreement_validator_mutual_entailment': abductive_agreement_validator_mutual_entailment,
            'consanguinity_threshold_validator_threshold': consanguinity_threshold_validator_threshold,
            'max_steps': max_steps,
            'torch_device': torch_device,
            'all_one_premise': all_one_premise,
            'mix_distractors': mix_distractors,
            'job_idx': idx
        }

        searches.append(search_args)

    if len(torch_devices) == 1:
        results = [__search__(**searches[0])]
    else:
        with Pool(len(torch_devices)) as p:
            results = p.map(__process_wrapper__, searches)

    searched_trees = []
    for result in results:
        searched_trees.extend(result)

    # EXPORT EVALUATIONS.
    with output_file.open('w') as f:
        json.dump([x.to_json() for x in searched_trees], f)

    if not silent:
        print("Finished Search.")


def __search__(
    trees: List[Tree],
    heuristic: str,
    step_types: List[str],
    step_model_names: List[str],
    nums_return_sequences: List[int],
    termination_criteria: List[str],
    validator_names: List[str],
    forward_agreement_validator_forward_model_name: str,
    forward_agreement_validator_entailment_model_name: str,
    forward_agreement_validator_agreement_threshold: float,
    forward_agreement_validator_mutual_entailment: bool,
    abductive_agreement_validator_abductive_model_name: str,
    abductive_agreement_validator_entailment_model_name: str,
    abductive_agreement_validator_agreement_threshold: float,
    abductive_agreement_validator_invalid_input_tolerance: int,
    abductive_agreement_validator_mutual_entailment: bool,
    consanguinity_threshold_validator_threshold: int,
    max_steps: int,
    torch_device: str,
    all_one_premise: bool = False,
    mix_distractors: bool = False,
    job_idx: int = 0,
) -> List[Tree]:
    """
    Wrapper for performing search on a file of trees.

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param heuristic: Name of the heuristic to score the fringe with
    :param step_types: Step model types you want to run, separate multiple types with a space
    :param step_model_names: Name of each pretrained step model to use for the corresponding step type
        (should be same length as step_types)
    :param nums_return_sequences: Number of returned generations per step model (if just 1 value, that value is shared
        across all step models)
    :param termination_criteria: Name of the termination criteria for stopping the search early (separate multiple with
        spaces
    :param validator_names: Name of the validators to use when step models generate new outputs. Validators will
        remove outputs it finds invalid before they populate the search fringe.
    :param forward_agreement_validator_forward_model_name Name of the forward model to use for the forward model
        agreement validator
    :param forward_agreement_validator_entailment_model_name Name of the entailment model to use for the forward model
        agreement validator
    :param forward_agreement_validator_agreement_threshold Threshold to use for the entailment score in the forward
        model agreement validator
    :param forward_agreement_validator_mutual_entailment For each generation, it checks entailment with the goal for
        the forward agreement validator essentially ensuring mutual entailment
    :param abductive_agreement_validator_abductive_model_name Name of the abductive model to use for the abductive model
        agreement validator
    :param abductive_agreement_validator_entailment_model_name Name of the entailment model to use for the abductive
        model agreement validator
    :param abductive_agreement_validator_agreement_threshold Threshold to use for the entailment score in the abductive
        model agreement validator
    :param abductive_agreement_validator_invalid_input_tolerance: The number of inputs that are allowed to be invalid
        in the abductive model agreement validator (i.e. number of abductive generations that can be below the agreement
        threshold).
    :param abductive_agreement_validator_mutual_entailment For each generation, it checks entailment with the goal for
        the forward agreement validator essentially ensuring mutual entailment
    :param consanguinity_threshold_validator_threshold hreshold for the maximum consanguinity (distance to first shared
        ancestor) of any two step inputs for a generation.
    :param max_steps: Number of maximum search steps for each example
    :param torch_device: Torch device to use for the inner search
    :param all_one_premise: Usually used for single shot models, takes a tree and combines all of it\'s premises into
        one.
    :param mix_distractors: If a tree has distractor premises, mix them in with the trees actual premises.
    :param job_idx: if the search is a part of a multiprocessed job, this is the job index.  This helps position the
        progress bars etc.
    :return: List of Tree objects that have expanded intermediates and hypotheses
    """

    torch.manual_seed(0)
    random.seed(0)

    if 'cuda' in torch_device and not torch.cuda.is_available():
        torch_device = 'cpu'

    if len(nums_return_sequences) == 1:
        nums_return_sequences = [nums_return_sequences[0]] * len(step_types)

    # LOAD UP STEP TYPES AND THEIR ASSOCIATED MODELS
    search_step_types = []
    for (step_type, model_name, num_return_sequences) in zip(step_types, step_model_names, nums_return_sequences):
        if step_type == 'abductive':
            abductive_model = AbductiveStepModel(
                model_name,
                device=torch.device(torch_device),
                num_return_sequences=num_return_sequences
            )

            search_step_types.append(
                AbductiveStepType(step_model=abductive_model)
            )
        elif step_type == 'forward':
            forward_model = ForwardStepModel(
                model_name,
                device=torch.device(torch_device),
                num_return_sequences=num_return_sequences
            )

            search_step_types.append(
                ForwardStepType(step_model=forward_model)
            )
        else:
            raise Exception(f"Specify a supported step_type. '{step_type}', is not supported yet.")

    # GET HEURISTIC SET UP
    if heuristic == 'BFS':
        search_heuristic = BreadthFirstSearchHeuristic
    elif heuristic == 'DFS':
        search_heuristic = DepthFirstSearchHeuristic
    elif heuristic == "StepTypeSwapBFS":
        search_heuristic = StepTypeSwapBFS
    elif heuristic == "StepTypeModeled":
        # TODO - change this to match your heuristic models names.
        # TODO - make this respect config values.
        search_heuristic = partial(StepTypeModeled, torch_device=torch_device, forward_name='forward_v3_gc', abductive_name='abductive_gc')
    else:
        raise Exception(f"Specify a supported heuristic. '{heuristic}', is not supported yet.")

    search_termination_criteria = []
    # GET TERMINATION CRITERIA SET UP
    if 'ExactMatchTermination' in termination_criteria:
        search_termination_criteria.append(ExactMatchTermination())
    if 'HypothesisAndIntermediateEntailment' in termination_criteria:
        search_termination_criteria.append(HypothesisAndIntermediateEntailment(torch_device=torch.device(torch_device)))

    validators = []
    if 'GeneratedInput' in validator_names:
        validators.append(GeneratedInputValidator())
    if 'ForwardAgreement' in validator_names:
        validators.append(ForwardAgreementValidator(
            forward_step_model_name=forward_agreement_validator_forward_model_name,
            entailment_model_name=forward_agreement_validator_entailment_model_name,
            agreement_threshold=forward_agreement_validator_agreement_threshold,
            mutual_entailment=forward_agreement_validator_mutual_entailment,
            torch_device=torch_device
        ))
    if 'AbductiveAgreement' in validator_names:
        validators.append(AbductiveAgreementValidator(
            abductive_step_model_name=abductive_agreement_validator_abductive_model_name,
            entailment_model_name=abductive_agreement_validator_entailment_model_name,
            agreement_threshold=abductive_agreement_validator_agreement_threshold,
            invalid_input_tolerance=abductive_agreement_validator_invalid_input_tolerance,
            mutual_entailment=abductive_agreement_validator_mutual_entailment,
            torch_device=torch_device
        ))
    if 'ConsanguinityThreshold' in validator_names:
        validators.append(ConsanguinityThresholdValidator(
            threshold=consanguinity_threshold_validator_threshold
        ))

    # BUILD THE SEARCH OBJECT
    search = Search(
        step_types=search_step_types,
        termination_criteria=search_termination_criteria,
        max_steps=max_steps
    )

    # PROGRESS BAR KWARGS FOR THE SEARCH FUNC.
    inner_search_pbar_kwargs = {
        'total': max_steps,
        'desc': f"Searching the fringe",
        'leave': False,
        'position': job_idx * 2 + 1
    }

    predicted_trees: List[Tree] = []

    # LOOP OVER TREES (MAIN LOOP)
    for tree in tqdm(
            trees,
            desc=f'Searching Trees on device {torch_device}',
            total=len(trees),
            position=job_idx * 2,
            leave=False,
    ):
        if mix_distractors:
            premises = [x for x in [*tree.premises, *tree.distractor_premises]]
            random.shuffle(premises)
        else:
            premises = tree.premises

        if all_one_premise:
            premises = [" ".join(premises)]

        original_premises = deepcopy(premises)

        # Perform the actual search for the tree (trying to get that missing premise)
        predicted_tree = search.search(
            premises=premises,
            hypotheses=tree.hypotheses,
            goal=tree.goal,
            heuristic=search_heuristic,
            validators=validators,
            show_progress=True,
            pbar_kwargs=inner_search_pbar_kwargs
        )

        predicted_tree.missing_premises = tree.missing_premises
        predicted_tree.__original_premises__ = original_premises

        predicted_trees.append(predicted_tree)

    return predicted_trees


if __name__ == "__main__":
    # SET UP PARAMETERS
    argparser = ArgumentParser()

    argparser.add_argument('--yaml_file', '-y', type=str, help='Path to a yaml config file')
    argparser.add_argument('--yaml_scopes', '-ys', type=str, nargs='+',
                           help='The objects/scopes in the yaml file. (i.e. abductive_only.search)')

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in')
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--heuristic', '-he', type=str,
                           choices=['BFS', 'DFS', 'StepTypeSwapBFS', 'StepTypeModeled'],
                           help='Name of the heuristic to score the fringe with')
    argparser.add_argument('--step_types', '-s', type=str, nargs='+', choices=['abductive', 'forward'],
                           help='Step model types you want to run, separate multiple types with a space')
    argparser.add_argument('--step_model_names', '-sm', type=str, nargs='+',
                           help='Name of each pretrained step model to use for the corresponding step type '
                                '(should be same length as --step_types)')
    argparser.add_argument('--nums_return_sequences', '-nrs', type=int, nargs='+',
                           help='Number of returned generations per step model (if just 1 value, that value is shared '
                                'across all step models)')
    argparser.add_argument('--termination_criteria', '-t', type=str, nargs='+',
                           choices=['HypothesisAndIntermediateEntailment', 'ExactMatchTermination'],
                           help='Name of the termination criteria for stopping the search early (separate multiple with'
                                'spaces)')

    argparser.add_argument('--validators', '-v', type=str, nargs='+',
                           help='Name of the validators to use when step models generate new outputs. Validators will'
                                ' remove outputs it finds invalid before they populate the search fringe.')
    argparser.add_argument('--forward_agreement_validator_forward_model_name', '-favfmn', type=str,
                           help='Name of the forward model to use for the forward model agreement validator')
    argparser.add_argument('--forward_agreement_validator_entailment_model_name', '-favfmn', type=str,
                           help='Name of the entailment model to use for the forward model agreement validator')
    argparser.add_argument('--forward_agreement_validator_agreement_threshold', '-favat', type=float,
                           help='Threshold to use for the entailment score in the forward model agreement validator')
    argparser.add_argument('--forward_agreement_validator_mutual_entailment', '-favme', action='store_true',
                           help='Looks at entailment going both ways for validating generations with the forward '
                                'model')

    argparser.add_argument('--abductive_agreement_validator_abductive_model_name', '-aavamn', type=str,
                           help='Name of the abductive model to use for the abductive model agreement validator')
    argparser.add_argument('--abductive_agreement_validator_entailment_model_name', '-aavfmn', type=str,
                           help='Name of the entailment model to use for the abductive model agreement validator')
    argparser.add_argument('--abductive_agreement_validator_agreement_threshold', '-aavat', type=float,
                           help='Threshold to use for the entailment score in the abductive model agreement validator')
    argparser.add_argument('--abductive_agreement_validator_invalid_input_tolerance', '-aaviit', type=int,
                           help='The number of inputs that are allowed to be invalid in the abductive model agreement '
                                'validator (i.e. number of abductive generations that can be below the agreement '
                                'threshold).')
    argparser.add_argument('--abductive_agreement_validator_mutual_entailment', '-aavme', action='store_true',
                           help='Looks at entailment going both ways for validating generations with the abductive '
                                'model')

    argparser.add_argument('--consanguinity_threshold_validator_threshold', '-ctvt', type=int,
                           help='Threshold for the maximum consanguinity (distance to first shared ancestor) of any '
                                'two step inputs for a generation.')

    argparser.add_argument('--max_steps', '-ms', type=int, help='Number of maximum search steps for each example')
    argparser.add_argument('--torch_devices', '-td', type=str, nargs='+',
                           help='Torch devices to use for the inner search, if multiple devices given the trees will be'
                                ' split across them then merged back into a single output file.')

    argparser.add_argument('--all_one_premise', '-aop', action='store_true',
                           help='Usually used for single shot models, takes a tree and combines all of it\'s premises '
                                'into one.')
    argparser.add_argument('--mix_distractors', '-md', action='store_true',
                           help='Mix trees with distractors into the premise set.')

    args = argparser.parse_args()

    yaml_file: Optional[Path] = Path(args.yaml_file) if args.yaml_file is not None else None
    yaml_scopes: Optional[str] = args.yaml_scopes

    if yaml_file:
        args = merge_yaml_and_namespace(
            yaml_file,
            args,
            scopes=yaml_scopes,
            default_scope='default.search',
            favor_namespace=True
        )

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _heuristic: str = args.heuristic
    _step_types: List[str] = args.step_types
    _step_model_names: List[str] = args.step_model_names
    _nums_return_sequences: List[int] = args.nums_return_sequences
    _termination_criteria: List[str] = args.termination_criteria
    _validator_names: List[str] = args.validators
    _forward_agreement_validator_forward_model_name: str = args.forward_agreement_validator_forward_model_name
    _forward_agreement_validator_entailment_model_name: str = args.forward_agreement_validator_entailment_model_name
    _forward_agreement_validator_agreement_threshold: float = args.forward_agreement_validator_agreement_threshold
    _forward_agreement_validator_mutual_entailment: bool = args.forward_agreement_validator_mutual_entailment
    _abductive_agreement_validator_forward_model_name: str = args.abductive_agreement_validator_forward_model_name
    _abductive_agreement_validator_entailment_model_name: str = args.abductive_agreement_validator_entailment_model_name
    _abductive_agreement_validator_agreement_threshold: float = args.abductive_agreement_validator_agreement_threshold
    _abductive_agreement_validator_invalid_input_tolerance: int = args.abductive_agreement_validator_invalid_input_tolerance
    _abductive_agreement_validator_mutual_entailment: bool = args.abductive_agreement_validator_mutual_entailment
    _consanguinity_threshold_validator_threshold: int = args.consanguinity_threshold_validator_threshold
    _max_steps: int = args.max_steps
    _torch_devices: List[str] = args.torch_devices
    _all_one_premise: bool = args.all_one_premise
    _mix_distractors: bool = args.mix_distractors

    search(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        heuristic=_heuristic,
        step_types=_step_types,
        step_model_names=_step_model_names,
        nums_return_sequences=_nums_return_sequences,
        termination_criteria=_termination_criteria,
        validator_names=_validator_names,
        forward_agreement_validator_forward_model_name=_forward_agreement_validator_forward_model_name,
        forward_agreement_validator_entailment_model_name=_forward_agreement_validator_entailment_model_name,
        forward_agreement_validator_agreement_threshold=_forward_agreement_validator_agreement_threshold,
        forward_agreement_validator_mutual_entailment=_forward_agreement_validator_mutual_entailment,
        abductive_agreement_validator_abductive_model_name=_abductive_agreement_validator_abductive_model_name,
        abductive_agreement_validator_entailment_model_name=_abductive_agreement_validator_entailment_model_name,
        abductive_agreement_validator_agreement_threshold=_abductive_agreement_validator_agreement_threshold,
        abductive_agreement_validator_invalid_input_tolerance=_abductive_agreement_validator_invalid_input_tolerance,
        abductive_agreement_validator_mutual_entailment=_abductive_agreement_validator_mutual_entailment,
        consanguinity_threshold_validator_threshold=_consanguinity_threshold_validator_threshold,
        max_steps=_max_steps,
        torch_devices=_torch_devices,
        all_one_premise=_all_one_premise,
        mix_distractors=_mix_distractors
    )
