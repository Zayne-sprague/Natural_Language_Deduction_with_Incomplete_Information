from search.tree.tree import Tree
from utils.paths import SEARCH_DATA_FOLDER

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict
import csv

import json

if __name__ == "__main__":
    # SET UP PARAMETERS
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over',
                           default=SEARCH_DATA_FOLDER / 'moral40_full.json')
    argparser.add_argument('--filter_file', '-ff', type=str, help='File that will be used to filter the input file',
                           required=True)
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--input_key_column', '-ikc', type=str, choices=['goal', 'missing_premises'],
                           default='missing_premises', help='Which part of the tree should we match one')
    argparser.add_argument('--filter_key_column', '-fkc', type=str, default='Goal',
                           help='Which column in the filter csv file should be used to filter on the input_key_column')

    args = argparser.parse_args()

    input_file: Path = Path(args.input_file)
    filter_file: Path = Path(args.filter_file)
    output_file: Path = Path(args.output_file)
    force_output: bool = args.force_output
    input_key_column: str = args.input_key_column
    filter_key_column: str = args.filter_key_column

    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json or jsonl file with an array of trees to run the search over.'
    assert filter_file.is_file() and str(filter_file).endswith('.csv'), \
        'Please specify a correct path to a csv file for filtering the input file on.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    filter_keys = []
    with filter_file.open('r') as f:
        reader = csv.DictReader(f)
        for line in reader:
            filter_keys.append(line[filter_key_column])

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_trees = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_trees = json.load(input_file.open('r'))
    trees = [Tree.from_json(t) for t in json_trees]
    filtered_trees = []

    for tree in trees:
        if input_key_column == 'goal' and tree.goal in filter_keys:
            filtered_trees.append(tree)
        elif input_key_column == 'missing_premises' and any([x in filter_keys for x in tree.missing_premises]):
            filtered_trees.append(tree)

    # Save Converted Trees.
    with output_file.open('w') as f:
        json.dump([x.to_json() for x in filtered_trees], f)

    print(f"Filtered {len(trees)} trees to {len(filtered_trees)} trees.  Saved at {output_file}")
