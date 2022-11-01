from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
from typing import List
import torch
from multiprocessing import Pool

from utils.paths import DATA_FOLDER
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from search.evaluation.metrics.entailement_metric import EntailmentEvaluation
from search.step_type import StepModel, ForwardStepModel


def __process_wrapper__(kwargs):
    return __score_proofs__(**kwargs)


def score_proofs(
        input_file: Path,
        output_file: Path,
        force_output: bool,
        score_methods: List[str],
        forward_step_model_name: str,
        torch_devices: str,
        silent: bool = False
) -> List[List[Tree]]:
    """
    Wrapper function for __score_proofs__

    Specifically, this wrapper helps spread the search across multiple devices by splitting the number of trees to
    search over.

    This splitting is done via multiprocessing and putting each instance of the search on a new core. (might not work if
    you have more devices than cpu cores... but lucky you if that's the case :D )

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param score_methods: What to score and save in the output (should have 1 value for each score_method)
    :param forward_step_model_name: The name of the folder to use for the Forward Step model
    :param torch_devices: Torch device to use for the inner search
    :param silent: No log messages
    :return: List of SubLists of proofs where each proof is now scored and the list of proofs is sorted with the best
        scoring proof at the first index.
    """

    # VALIDATE ARGS
    assert input_file.is_file() and str(input_file).endswith('.json'), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'
    assert forward_step_model_name is not None or 'forward' not in score_methods, \
        'Please specify the forward step model name or do not score using the forward score method.'

    # LOAD UP TREES
    json_proofs = json.load(input_file.open('r'))
    all_trees: List[List[Tree]] = [[Tree.from_json(t) for t in p] for p in json_proofs]

    device_trees = []

    trees_per_device = len(all_trees) / len(torch_devices)
    for i in range(len(torch_devices)):
        start_idx = int(min(i * trees_per_device, len(all_trees) - 1))
        end_idx = int(min((i + 1) * trees_per_device, len(all_trees)))

        if i == len(torch_devices) - 1:
            # If its the last device, always take all the trees up till the last index.
            end_idx = len(all_trees)

        device_trees.append(all_trees[start_idx:end_idx])

    score_calls = []

    for idx, (torch_device, trees) in enumerate(zip(torch_devices, device_trees)):
        score_args = {
            'trees': trees,
            'score_methods': score_methods,
            'forward_step_model_name': forward_step_model_name,
            'torch_device': torch_device,
            'job_idx': idx
        }

        score_calls.append(score_args)

    if len(torch_devices) == 1:
        results = [__score_proofs__(**score_calls[0])]
    else:
        with Pool(len(torch_devices)) as p:
            results = p.map(__process_wrapper__, score_calls)

    proofs = []
    for result in results:
        proofs.extend(result)

    if not silent:
        print(f'Found proofs for {len(proofs)} trees. All proofs are saved at {str(output_file)}')

    with output_file.open('w') as f:
        json.dump([[x.to_json() for x in p] for p in proofs], f)

    return proofs


def __score_proofs__(
        trees: List[List[Tree]],
        score_methods: List[str],
        forward_step_model_name: str,
        torch_device: str,
        job_idx: int = 0
) -> List[List[Tree]]:
    """
    Given a file that contains a list of lists where each sub list contains proofs for a specific tree, this func will
    score each proof in the sublist then sort the proofs in the sublist so that the best proofs are at the beginning of
    the list.

    :param Trees: List of trees with proofs we want to rank
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param score_methods: What to score and save in the output (should have 1 value for each score_method)
    :param forward_step_model_name: The name of the folder to use for the Forward Step model
    :param torch_device: Torch device to use for the inner search
    :param silent: No log messages
    :return: List of SubLists of proofs where each proof is now scored and the list of proofs is sorted with the best
        scoring proof at the first index.
    """

    if 'cuda' in torch_device and not torch.cuda.is_available():
        torch_device = 'cpu'

    forward_model = None
    entailment_model = None
    if forward_step_model_name:
        forward_model = ForwardStepModel(forward_step_model_name, device=torch.device(torch_device))
        entailment_model = EntailmentEvaluation('wanli_entailment_model', torch_device=torch.device(torch_device))

    proofs = []

    for tree in tqdm(
            trees,
            total=len(trees),
            desc=f"Scoring Proofs on device {torch_device}",
            position=job_idx,
            leave=False
    ):
        scored_proofs = []

        for proof in tree:

            proof_scores = {k: 0 for k in score_methods}


            if 'forward_agreement' in score_methods:

                forward_inputs = [
                    forward_model.format(proof, intermediate.inputs) for intermediate in proof.intermediates
                ]
                goals = [intermediate.output for intermediate in proof.intermediates]

                generations = forward_model.sample(forward_inputs)

                entailment_inputs = []
                entailment_goals = []

                for generation, goal in zip(generations, goals):
                    if isinstance(generations, str):
                        entailment_inputs.append(generations)
                        entailment_goals.append(goal)
                    else:
                        entailment_inputs.extend(generation)
                        entailment_goals.extend([goal] * len(generation))

                escores = entailment_model.score(entailment_goals, entailment_inputs)

                escores = [
                    escores[i:i + forward_model.num_return_sequences]
                    for i in range(0, len(escores), forward_model.num_return_sequences)
                ]

                for idx, intermediate in enumerate(proof.intermediates):
                    score = sum(escores[idx])
                    proof_scores['forward_agreement'] += score
                    intermediate.scores['forward_agreement'] = score

                proof_scores['forward_agreement'] /= len(proof.intermediates)

            proof_score = sum([v for k, v in proof_scores.items()]) / len(score_methods)

            scored_proofs.append((proof, proof_score))

        scored_proofs = list(sorted(scored_proofs, key=lambda x: x[1], reverse=True))
        proofs.append([x[0] for x in scored_proofs])

    return proofs


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over')
    argparser.add_argument('--output_file', '-o', type=str, required=True,
                           help='Path to the output file that the evaluations will be stored in')
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--score_methods', '-st', type=str, nargs="+", required=True,
                           choices=["forward_agreement"],
                           help='What to score and save in the output (separate with space, should have 1 value for '
                                'each score_method)')
    argparser.add_argument('--forward_step_model_name', '-fsm', type=str,
                           help='The name of the folder to use for the Forward Step model')
    argparser.add_argument('--torch_devices', '-td', type=str, nargs='+', default=['cpu'],
                           help='Torch devices to use for the inner search, if multiple devices given the trees will be'
                                ' split across them then merged back into a single output file.')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _score_methods: List[str] = args.score_methods
    _forward_step_model_name: str = args.forward_step_model_name
    _torch_devices: str = args.torch_devices

    score_proofs(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        score_methods=_score_methods,
        forward_step_model_name=_forward_step_model_name,
        torch_devices=_torch_devices
    )
