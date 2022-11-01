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

def proof_reports(
        input_files: List[Path],
        input_experiments: List[Path],
):
    """
    Wrapper that can export a list of trees from a file or a list of trees where each tree is a list of proofs for that
    tree.

    :param input_files: Paths to files with trees to search over
    :param input_experiments: Paths to experiment folders with searches
    """

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
        scored_searches = experiment / 'data'

        for file in scored_searches.glob('shallow_trees*.json'):
            tree_sets.append(json.load(file.open('r')))
            file_names.append(file)

    if len(tree_sets) == 0:
        return

    for idx, tree_set in enumerate(tree_sets):
        trees: List[Tree] = [Tree.from_json(x) for x in tree_set]
        filename = file_names[idx]

        avg_len = sum([len(x) for x in trees]) / len(trees)
        print(avg_len)

        avg_depth = sum([x.get_depth() for x in trees]) / len(trees)
        print(avg_depth)



if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_files', '-i', type=str, nargs='+', default=[],
                           help='Path to file with trees to search over')
    argparser.add_argument('--input_experiments', '-e', type=str, nargs='+', default=[],
                           help='Path to experiment folders that contain the searched trees.')

    args = argparser.parse_args()

    _input_files: List[Path] = [Path(x) for x in args.input_files]
    _input_experiments: List[Path] = [SEARCH_OUTPUT_FOLDER / x for x in args.input_experiments]

    proof_reports(
        input_files=_input_files,
        input_experiments=_input_experiments,
    )
