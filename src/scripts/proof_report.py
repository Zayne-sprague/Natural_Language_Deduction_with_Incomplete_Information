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


def __avg_num_proofs__(
        proofs: List[List[Tree]]
):
    total = 0
    for proof_set in proofs:
        total += len(proof_set)

    return total / len(proofs)


def __avg_tree_length__(
        proofs: List[List[Tree]]
):
    total_averages = 0
    for proof_set in proofs:
        total_lengths = 0
        for tree in proof_set:
            total_lengths += len(tree)
        total_averages += total_lengths / len(proof_set)

    return total_averages / len(proofs)


def __avg_forward_agreement__(
        proofs: List[List[Tree]]
):
    total_agreement = 0
    for proof_set in proofs:
        total_set_agreement = 0
        for tree in proof_set:
            total_tree_agreement = 0
            for intermediate in tree.intermediates:
                total_tree_agreement += intermediate.scores.get('forward_agreement', 0)
            total_set_agreement += total_tree_agreement / len(tree)
        total_agreement += total_set_agreement / len(proof_set)
    return total_agreement / len(proofs)


def __avg_number_of_premises_used__(
        proofs: List[List[Tree]]
):
    total = 0
    for proof_set in proofs:
        total_set_premise_count = 0
        for tree in proof_set:

            total_orig_premises = len(tree.__original_premises__)
            orig_premises = {x for x in tree.__original_premises__}

            for intermediate in tree.intermediates:
                for arg in intermediate.inputs:
                    val = tree.get_step_value(arg)
                    if val in orig_premises:
                        orig_premises.remove(val)

            total_premises_used = total_orig_premises - len(orig_premises)

            total_set_premise_count += total_premises_used / total_orig_premises

        total += total_set_premise_count / len(proof_set)
    return total / len(proofs)

def __proof_report__(
        proofs: List[List[Tree]]
):

    average_num_of_proofs = __avg_num_proofs__(proofs)
    average_tree_len = __avg_tree_length__(proofs)
    average_forward_agreement = __avg_forward_agreement__(proofs)
    average_premises_used = __avg_number_of_premises_used__(proofs)

    print(f"Avg Number of Proofs:  {average_num_of_proofs:.02f}")
    print(f"Avg Tree Length:       {average_tree_len:.02f}")
    print(f"Avg Forward Agreement: {average_forward_agreement:.02f}")
    print(f"Avg Premises Used:     {average_premises_used:.02f}")



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
        scored_searches = experiment / 'output'

        for file in scored_searches.glob('scored_tree_proofs*.json'):
            tree_sets.append(json.load(file.open('r')))
            file_names.append(file)

    if len(tree_sets) == 0:
        return

    for idx, tree_set in enumerate(tree_sets):
        proofs: List[List[Tree]] = [[Tree.from_json(y) for y in x] for x in tree_set]
        filename = file_names[idx]

        print(f" === Report for {filename} ===")
        __proof_report__(proofs)
        print(" === === === === === ===")


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_files', '-i', type=str, nargs='+', default=[],
                           help='Path to file with trees to search over')
    argparser.add_argument('--input_experiments', '-e', type=str, nargs='+',
                           help='Path to experiment folders that contain the searched trees.')

    args = argparser.parse_args()

    _input_files: List[Path] = [Path(x) for x in args.input_files]
    _input_experiments: List[Path] = [SEARCH_OUTPUT_FOLDER / x for x in args.input_experiments]

    proof_reports(
        input_files=_input_files,
        input_experiments=_input_experiments,
    )
