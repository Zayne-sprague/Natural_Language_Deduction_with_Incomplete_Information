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


def __convert_to_coverage_csv__(
        output_file: Path,
        trees_with_filenames: List[Tuple[Tree, str]],
):

    CSV_FIELDNAMES = [
        'File Name',
        'Inputs',
        'Generated Premise',
        'Goal Premise'
    ]

    with output_file.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        rows = []
        for tree_idx, (tree, filename) in enumerate(trees_with_filenames):
            missing_premise = tree.missing_premises[0]
            recovered_step = [x for x in tree.intermediates if x.tags.get('step_type') is None][0]
            inputs = [tree.get_step_value(recovered_step.inputs[0]), recovered_step.output]
            output = tree.get_step_value(recovered_step.inputs[1])

            if missing_premise.endswith('.'):
                missing_premise = missing_premise[:-1]
            if output.endswith('.'):
                output = output[:-1]

            if missing_premise in output:
                missing_premise = "*" + missing_premise
            if output in missing_premise:
                output = "*" + output

            output += '.'
            missing_premise += '.'

            inputs_str = " ".join(inputs)

            row = {
                "File Name": filename,
                'Inputs': inputs_str,
                'Generated Premise': output,
                'Goal Premise': missing_premise,
            }

            rows.append(row)

        writer.writeheader()
        writer.writerows(rows)


def convert_to_step_validity_csv(
        input_files: List[Path],
        input_experiments: List[Path],
        output_file: Path,
        force_output: bool
):
    """
    Wrapper that can export a list of trees from a file or a list of trees where each tree is a list of proofs for that
    tree.

    :param input_files: Paths to files with trees to search over
    :param input_experiments: Paths to experiment folders with searches
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    """

    # VALIDATE ARGS
    assert (not output_file.exists() or force_output) and str(output_file).endswith('.json'), \
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

        # for file in scored_searches.glob('scored_tree_proofs*.json'):
        for file in scored_searches.glob('enhanced_proofs*'):
            tree_sets.append(json.load(file.open('r')))
            file_names.append(file)

    if len(tree_sets) == 0:
        return

    for idx, tree_set in enumerate(tree_sets):
        top_proofs = [x[0] for x in tree_set]

        for idx, proof in enumerate(top_proofs):
            proof = Tree.from_json(proof)
            for intermediate in proof.intermediates:
                step_type = intermediate.tags.get('step_type')
                if not step_type:
                    step_type = 'abductive'
                elif step_type == 'bridge':
                    step_type = 'abductive'
                intermediate.tags['step_type'] = step_type

            top_proofs[idx] = proof.to_json()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w') as f:
            json.dump(top_proofs, f)

if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_files', '-i', type=str, nargs='+', default=[],
                           help='Path to file with trees to search over')
    argparser.add_argument('--input_experiments', '-e', type=str, nargs='+',
                           help='Path to experiment folders that contain the searched trees.')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')

    args = argparser.parse_args()

    _input_files: List[Path] = [Path(x) for x in args.input_files]
    _input_experiments: List[Path] = [SEARCH_OUTPUT_FOLDER / x for x in args.input_experiments]
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output

    convert_to_step_validity_csv(
        input_files=_input_files,
        input_experiments=_input_experiments,
        output_file=_output_file,
        force_output=_force_output
    )
