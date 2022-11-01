import copy
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm

from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration


def create_shallow_trees(
        input_file: Path,
        output_file: Path,
        force_output: bool = False,
        depth: int = 2,
        min_depth: int = 2,
        max_depth: int = 2,
        allow_small_trees: bool = False,
        silent: bool = False,
) -> List[Tree]:
    """
    Given a file containing a list of trees, convert those trees into a specific depth.

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param depth: Depth to make the new trees. (default is 2)
    :param min_depth: Only allow trees greater than or equal to this value (useful with allow_small_trees)
    :param allow_small_trees: If an entire tree is smaller than the specified depth, keep the full tree.
    :param silent: No log messages
    :return: A list of Tree objects that are of the specified depth
    """

    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_trees = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_trees = json.load(input_file.open('r'))
    trees = [Tree.from_json(t) for t in json_trees]

    shallow_trees: List[Tree] = []

    # Slice trees.
    for tree in tqdm(trees, desc='Slicing Trees', total=len(trees), position=0, leave=False):

        pure_tree = tree.slice(len(tree.intermediates)-1, depth=-1)

        if depth > -1:
            subtrees = [
                tree.slice(x, depth=depth) for x in range(len(tree.intermediates))
            ]
        else:
            tree.reduce_to_intermediate_subtree(len(tree.intermediates) - 1)
            subtrees = [tree]

        accepted_subtrees = []
        for subtree in subtrees:
            subtree_depth = subtree.get_depth()

            if depth == -1:
                accepted_subtrees.append(subtree)
            elif subtree_depth < depth and allow_small_trees and subtree_depth >= min_depth:
                # If the size of the desired subtrees is larger than the actual tree, just search over the tree.
                # If the specified size was -1 then that means to search the entire tree as well.
                accepted_subtrees.append(subtree)
            elif subtree_depth >= depth and subtree_depth <= max_depth:
                accepted_subtrees.append(subtree)

        for subtree in accepted_subtrees:
            subtree.distractor_premises = [x for x in tree.premises if x not in pure_tree.premises]

        shallow_trees.extend(accepted_subtrees)

    # Save Sliced Trees.
    with output_file.open('w') as f:
        json.dump([x.to_json() for x in shallow_trees], f)

    if not silent:
        print(f"{len(shallow_trees)} shallow trees created and saved in {output_file}.  Exiting...")

    return shallow_trees


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--depth', '-d', type=int,
                           help='Depth to make the new trees. (default is 2)',
                           default=2)
    argparser.add_argument('--min_depth', '-md', type=int,
                           help='Only allow trees greater than or equal to this value (useful inconjunction with -a)',
                           default=2)
    argparser.add_argument('--max_depth', '-mxd', type=int,
                           help='Only allow trees less than or equal to this value',
                           default=2)
    argparser.add_argument('--allow_small_trees', '-a', action='store_true', dest='allow_small_trees',
                          help='If an entire tree is smaller than the specified depth, keep the full tree.')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _depth: int = args.depth
    _min_depth: int = args.min_depth
    _max_depth: int = args.max_depth
    _allow_small_trees: bool = args.allow_small_trees

    create_shallow_trees(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        depth=_depth,
        min_depth=_min_depth,
        max_depth=_max_depth,
        allow_small_trees=_allow_small_trees
    )
