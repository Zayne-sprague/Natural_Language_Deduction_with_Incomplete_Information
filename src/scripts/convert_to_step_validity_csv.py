from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm
import csv
import random
random.seed(1)

from utils.paths import SEARCH_DATA_FOLDER
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from utils.paths import SEARCH_OUTPUT_FOLDER


def __convert_to_step_validity_csv__(
        output_file: Path,
        steps_with_filenames: List[Tuple[StepGeneration, str, Tree]],
        use_proofs: bool = False
):

    CSV_FIELDNAMES = [
        'Inputs',
        'Conclusion',
        'Step Type',
        'File Name'
    ]

    with output_file.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        rows = []
        for tree_idx, (step, filename, tree) in enumerate(steps_with_filenames):
            step_type = step.tags['step_type']

            if step_type == 'intermediate' or use_proofs:
                inputs = " ".join([tree.get_step_value(x) for x in step.inputs])
                conclusion = step.output
            else:
                inputs = " ".join([tree.get_step_value(step.inputs[0]), step.output])
                conclusion = tree.get_step_value(step.inputs[1])

            row = {
                'Inputs': inputs,
                'Conclusion': conclusion,
                'Step Type': step.tags['step_type'],
                "File Name": filename
            }

            rows.append(row)

        writer.writeheader()
        writer.writerows(rows)


def convert_to_step_validity_csv(
        input_files: List[Path],
        input_experiments: List[Path],
        output_file: Path,
        sample_amount: int,
        use_proofs: bool,
        force_output: bool
):
    """
    Wrapper that can export a list of trees from a file or a list of trees where each tree is a list of proofs for that
    tree.

    :param input_files: Paths to files with trees to search over
    :param input_experiments: Paths to experiment folders with searches
    :param output_file: Path to the output file that the evaluations will be stored in
    :param sample_amount: Number of steps to pull from each search
    :param force_output: Overwrite anything currently written in the output file path.
    """

    # VALIDATE ARGS
    assert (not output_file.exists() or force_output) and str(output_file).endswith('.csv'), \
        'Please specify an empty csv file path for the output parameter -o OR specify the force flag -f'

    tree_sets = []
    file_names = []

    for input_file in input_files:
        # LOAD UP TREES
        if str(input_file).endswith('.jsonl'):
            tree_sets.append(list(jsonlines.open(str(input_file), 'r')))
            file_names.append(input_file)
        else:
            tree_sets.append(json.load(input_file.open('r')))
            file_names.append(input_file)

    for experiment in input_experiments:
        scored_searches = experiment / 'output'

        if use_proofs:
            for file in scored_searches.glob('scored_tree_proofs*.json'):
                tree_sets.append(json.load(file.open('r')))
                file_names.append(file)

        else:
            for file in scored_searches.glob('scored_search*.json'):
                tree_sets.append(json.load(file.open('r')))
                file_names.append(file)

    if len(tree_sets) == 0:
        return

    steps_with_filenames = []
    for idx, tree_set in enumerate(tree_sets):
        search_steps = []

        if use_proofs:
            proofs: List[List[Tree]] = [[Tree.from_json(y) for y in x] for x in tree_set]
            trees = [x[0] for x in proofs]
        else:
            trees: List[Tree] = [Tree.from_json(x) for x in tree_set]

        for tree in trees:
            if use_proofs:
                for intermediate in tree.intermediates:
                    intermediate.tags['step_type'] = 'forward' if intermediate.tags.get('step_type') == 'forward' else 'abductive'
            else:
                for intermediate in tree.intermediates:
                    intermediate.tags['step_type'] = 'intermediate'
                for hypothesis in tree.hypotheses:
                    hypothesis.tags['step_type'] = 'abduction'

            steps = tree.intermediates
            steps.extend(tree.hypotheses)

            search_steps.extend([(x, str(file_names[idx]), tree) for x in steps])

        steps_with_filenames.extend(random.sample(search_steps, min(sample_amount, len(search_steps))))

    random.shuffle(steps_with_filenames)

    __convert_to_step_validity_csv__(output_file, steps_with_filenames, use_proofs)


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_files', '-i', type=str, nargs='+', default=[],
                           help='Path to file with trees to search over')
    argparser.add_argument('--input_experiments', '-e', type=str, nargs='+',
                           help='Path to experiment folders that contain the searched trees.')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--sample_amount', '-sa', type=int, help='How many steps to sample from each search',
                           default=20)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--use_proofs', '-p', dest='use_proofs', action='store_true',
                           help='Use proofs and select from the top proof for sampling, not anywhere in the search'
                                ' state.')


    args = argparser.parse_args()

    _input_files: List[Path] = [Path(x) for x in args.input_files]
    _input_experiments: List[Path] = [SEARCH_OUTPUT_FOLDER / x for x in args.input_experiments]
    _output_file: Path = Path(args.output_file)
    _sample_amount: int = args.sample_amount
    _force_output: bool = args.force_output
    _use_proofs: bool = args.use_proofs

    convert_to_step_validity_csv(
        input_files=_input_files,
        input_experiments=_input_experiments,
        output_file=_output_file,
        sample_amount=_sample_amount,
        use_proofs=_use_proofs,
        force_output=_force_output
    )
