from argparse import ArgumentParser
from pathlib import Path
from typing import List
import json
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations

from search.tree.tree import Tree, StepGeneration


def create_premise_masked_trees(
        input_file: Path,
        output_file: Path,
        force_output: bool = False,
        premises_to_mask: int = 1,
        silent: bool = False
) -> List[Tree]:
    """
    Given a file which contains a list of trees, create variants of each tree where one of the premises is masked.

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param premises_to_mask: How many premises to mask (default is 1)
    :param silent: No log messages
    :return: List of Tree objects with a premise masked.
    """

    # VALIDATE ARGS
    assert input_file.is_file() and str(input_file).endswith('.json'), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    json_trees = json.load(input_file.open('r'))
    trees = [Tree.from_json(t) for t in json_trees]

    masked_trees: List[Tree] = []

    # Slice trees.
    for tree in tqdm(trees, desc='Masking Trees', total=len(trees), position=0, leave=False):
        premises = list(range(len(tree.premises)))

        if premises_to_mask == 0:
            masked_trees.append(tree)
            continue

        masks = combinations(premises, min([len(premises), premises_to_mask]))

        # For each mask, mask a unique premise we haven't masked before and store it as an example to run
        for mask in masks:
            masked_example = deepcopy(tree)

            # For each premise idx, mask the associated premise.  If we sort, we do not have to worry about premise
            # re-indexing (i.e. masking premise 1 will make all premises above it their idx minus 1.
            for premise_idx in sorted(mask, reverse=True):
                masked_example.mask_premise(premise_idx)

            masked_trees.append(masked_example)

    # Save Sliced Trees.
    with output_file.open('w') as f:
        json.dump([x.to_json() for x in masked_trees], f)

    if not silent:
        print(f"{len(masked_trees)} masked trees created and saved in {output_file}.  Exiting...")

    return masked_trees


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--premises_to_mask', '-p', type=int, default=1,
                           help='How many premises to mask (default is 1)')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _premises_to_mask: int = args.premises_to_mask

    create_premise_masked_trees(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        premises_to_mask=_premises_to_mask
    )
