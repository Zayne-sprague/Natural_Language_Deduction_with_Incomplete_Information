from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm
import csv
from copy import deepcopy
import random
random.seed(1)

from utils.paths import SEARCH_DATA_FOLDER
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from utils.paths import SEARCH_OUTPUT_FOLDER


def __convert_to_diff_csv__(
        output_file: Path,
        steps_with_filenames: List[List[Tuple[str, StepGeneration, Tree]]],
):

    example_step = steps_with_filenames[0]
    number_of_files = len(example_step)

    CSV_FIELDNAMES = [
        'Step Number',
        'File Name',
        'Inputs',
        'Generation',
        'Step Type',
    ]

    with output_file.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        rows = []
        for idx, step_group in enumerate(steps_with_filenames):
            for (filename, step, tree) in step_group:
                inputs = " ".join([tree.get_step_value(x) for x in step.inputs])
                output = step.output
                step_type = step.tags['step_type']

                row = {
                    'Step Number': idx,
                    'File Name': str(filename),
                    'Inputs': inputs,
                    'Generation': output,
                    'Step Type': step_type
                }

                rows.append(row)

        writer.writeheader()
        writer.writerows(rows)


def convert_to_diff_csv(
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

        for file in scored_searches.glob('scored_search*.json'):
            tree_sets.append(json.load(file.open('r')))
            file_names.append(file)

    if len(tree_sets) == 0:
        return

    for idx, tree_set in enumerate(tree_sets):
        trees = [Tree.from_json(x) for x in tree_set]
        tree_sets[idx] = trees

    # Sync up trees across files, drop those that do not match
    aligned_tree_table: Dict[str, List[Tuple[Tree, str]]] = {}
    for (tree_set, filename) in zip(tree_sets, file_names):
        for tree in tree_set:
            key = f'{tree.goal}|||{", ".join(tree.missing_premises)}'
            entry = aligned_tree_table.get(key, [])
            entry.append((filename, tree))
            aligned_tree_table[key] = entry

    tree_table: Dict[str, Tuple[List[Tree], str]] = {}
    for k, v in aligned_tree_table.items():
        if len(v) == len(file_names):
            tree_table[k] = v

    aligned_steps: List[List[Tuple[str,  StepGeneration, Tree]]] = []
    # With aligned trees, find steps with the same inputs
    for tree_key, trees_and_files in tree_table.items():
        aligned_tree_steps = {}
        for (filename, tree) in trees_and_files:
            tree_steps = {}

            for intermediate in tree.intermediates:
                key = "|||".join([tree.get_step_value(x) for x in intermediate.inputs])
                key = f"intermediate:{key}"
                groups = aligned_tree_steps.get(key, {})
                aligned_tree_steps[key] = groups

                steps = groups.get(str(filename), [])

                intermediate.tags['step_type'] = 'forward'

                steps.append((filename, intermediate, tree))

                aligned_tree_steps[key][str(filename)] = steps

            for hypothesis in tree.hypotheses:
                key = "|||".join([tree.get_step_value(x) for x in hypothesis.inputs])
                key = f"hypothesis:{key}"
                groups = aligned_tree_steps.get(key, {})
                aligned_tree_steps[key] = groups

                steps = groups.get(str(filename), [])

                hypothesis.tags['step_type'] = 'abductive'

                steps.append((filename, hypothesis, tree))

                aligned_tree_steps[key][str(filename)] = steps

        for k, v in aligned_tree_steps.items():
            if len(v) == len(file_names):
                aligned_steps.append(v)

    # Get Diffs
    diff_steps: List[List[Tuple[str, StepGeneration, Tree]]] = []
    for aligned_group in aligned_steps:
        generation_sets = [{x.output for (_, x, _) in g} for _, g in aligned_group.items()]

        unique_group_steps: List[str, StepGeneration, Tree] = []
        for idx, gen_set in enumerate(generation_sets):
            left_overs = deepcopy(gen_set)
            for oidx, other_set in enumerate(generation_sets):
                if idx == oidx:
                    continue
                left_overs -= other_set
            left_overs = list(left_overs)
            group_keys = list(aligned_group.keys())

            left_over_steps = [x for (_, x, _) in aligned_group[group_keys[idx]] if x.output in left_overs]

            unique_steps = [(aligned_group[group_keys[idx]][0][0], x, aligned_group[group_keys[idx]][0][2]) for x in left_over_steps]

            [unique_group_steps.append(x) for x in unique_steps]

        diff_steps.append(unique_group_steps)
    __convert_to_diff_csv__(output_file, diff_steps)


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

    convert_to_diff_csv(
        input_files=_input_files,
        input_experiments=_input_experiments,
        output_file=_output_file,
        force_output=_force_output
    )
