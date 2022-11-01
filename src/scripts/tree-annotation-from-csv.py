from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm
import csv

from utils.paths import SEARCH_DATA_FOLDER
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration


def find_and_set_step_generation_label(tree: Tree, label: str, label_type: str, generation: str):
    intermediates = tree.intermediates
    hypotheses = tree.hypotheses

    for idx, intermediate in enumerate(intermediates):
        if intermediate.output == generation:
            tree.intermediates[idx].annotations[label_type] = label.split(',')
            return tree

    for idx, hypothesis in enumerate(hypotheses):
        if hypothesis.output == generation:
            tree.hypotheses[idx].annotations[label_type] = label.split(',')
            return tree
    return tree


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, required=True,
                           help='Path to a tree scores json file that will be paired with the labeled csv')
    argparser.add_argument('--labeled_csv', '-l', type=str, required=True,
                           help='Path to a tree scores json file that will be paired with the labeled csv')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--tree_idx_column', '-idx', type=str, default='Tree Number',
                           help='Which column of the csv can be used as an index to grab the tree from the '
                                'input_tree_file')
    argparser.add_argument('--label_types', '-lt', type=str, nargs='+',
                           choices=['goal', 'missing_premises', 'valid_step'],
                           help='Which part of the tree is this annotation for (what is the label), i.e. an annotation'
                                'for the goal or for the missing premises.')
    argparser.add_argument('--csv_label_columns', '-clc', type=str, nargs='+',
                           help='Which column holds the label for the tree.')
    argparser.add_argument('--csv_generation_column', '-cgc', type=str,
                           help='Which column holds the step generations output (this is used to match the label to the'
                                'exact step generation).')
    argparser.add_argument('--label_delimiter', type=str, default=',',
                           help='For missing_premises you could have multiple labels, which means you need to separate'
                                'them somehow.  This allows you to set the param to delimit the labels in the same'
                                'column.')

    args = argparser.parse_args()

    input_file: Path = Path(args.input_file)
    label_file: Path = Path(args.labeled_csv)
    output_file: Path = Path(args.output_file)
    force_output: bool = args.force_output
    tree_idx_column: str = args.tree_idx_column
    label_types: List[str] = args.label_types
    csv_label_columns: List[str] = args.csv_label_columns
    csv_generation_column: str = args.csv_generation_column

    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty csv file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_trees = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_trees = json.load(input_file.open('r'))
    trees = [Tree.from_json(t) for t in json_trees]

    with label_file.open('r') as f:
        reader = csv.DictReader(f)

        for line in reader:

            for label_column, label_type in zip(csv_label_columns, label_types):
                tree_idx = line.get(tree_idx_column, None)
                label = line.get(label_column, None)
                generation = line.get(csv_generation_column, None)

                if not tree_idx or not label or not csv_generation_column or label == '':
                    continue

                tree_idx = int(tree_idx)
                tree = trees[tree_idx]
                trees[tree_idx] = find_and_set_step_generation_label(tree, label, label_type, generation)

    # Save Labeled Trees.
    with output_file.open('w') as f:
        json.dump([x.to_json() for x in trees], f)
