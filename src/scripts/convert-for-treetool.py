from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict
import json
from jsonlines import jsonlines
from tqdm import tqdm

from utils.paths import SEARCH_DATA_FOLDER
from search.tree.tree import Tree


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over',
                           default=SEARCH_DATA_FOLDER / 'moral40_full.json')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--include_missing_premises', '-imp', dest='include_missing_premises', action='store_true',
                           help='If there are missing premises in the tree, show them as orphaned premise statements'
                                ' with a prepended title "Missing Premise: ".')
    argparser.add_argument('--include_scores', '-is', nargs='+', type=str, choices=['goal', 'missing_premises'],
                           help='Scores you would like each generated step to display (separate with whitespace, '
                                'score will be displayed as a newline appended to the generation output.')

    args = argparser.parse_args()

    input_file: Path = Path(args.input_file)
    output_file: Path = Path(args.output_file)
    force_output: bool = args.force_output
    include_missing_premises: bool = args.include_missing_premises
    include_scores: List[str] = args.include_scores

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

    converted_trees: List[Dict[str, any]] = []

    for tree in trees:
        converted_tree = Tree.from_canonical_json(tree.to_canonical_json())

        if include_scores and len(include_scores) > 0:
            for idx, intermediate in enumerate(tree.intermediates):
                for score_type in include_scores:
                    score = intermediate.scores.get(score_type, None)

                    if isinstance(score, list) or isinstance(score, tuple):
                        score = ", ".join([f'{x:.3f}' for x in score])
                    else:
                        score = f'{score:.3f}'

                    converted_tree.intermediates[idx].output += f'| {score_type}: {score} |' if score else ''

        for idx, intermediate in enumerate(tree.intermediates):
            for in_idx, x in enumerate(intermediate.inputs):
                step_key = tree.get_step_key(x)
                step_value = tree.get_step_value(x)
                if step_key == 'm':
                    del converted_tree.intermediates[idx].inputs[in_idx]

        if include_missing_premises:
            converted_tree.premises.extend([f'Missing Premise: {x}' for x in tree.missing_premises])

        converted_trees.append(converted_tree.to_canonical_json())

    # Save Converted Trees.
    with output_file.open('w') as f:
        json.dump(converted_trees, f)

    print(f"{len(converted_trees)} trees converted for the treetool visualization saved in {output_file}.  Exiting...")
