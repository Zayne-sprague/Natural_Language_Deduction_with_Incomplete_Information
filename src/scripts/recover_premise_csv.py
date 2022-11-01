from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm
import csv

from utils.paths import DATA_FOLDER
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration


def convert_tree_to_missing_premise_csv(
        output_file: Path,
        trees: List[Tree],
        write_step_types: List[str],
        recover_threshold: float = 0.8,
        silent: bool = False
):

    CSV_FIELDNAMES = [
        'Tree Number',
        'Tree Recovered',
        'Step Recovered',
        'Score',
        'Generation',
        'Missing Premises',
        'Step Type',
        'Inputs',
        'Equation',
        'Missing Premise Scores',
        'Recover Threshold'
    ]

    with output_file.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        rows = []
        for tree_idx, tree in enumerate(trees):
            tree_recovered = False
            new_rows = []

            if 'intermediate' in write_step_types:
                for intermediate_idx, intermediate in enumerate(tree.intermediates):
                    mp_score = sum(intermediate.scores.get('missing_premises', [-1])) / max(len(intermediate.scores.get('missing_premises', [-1])), 1)
                    tree_recovered = True if mp_score >= recover_threshold else tree_recovered

                    row = {
                        'Tree Number': tree_idx,
                        'Tree Recovered': False,
                        'Step Recovered': mp_score >= recover_threshold,
                        'Score': mp_score,
                        'Generation': intermediate.output,
                        'Missing Premises': ' | '.join(tree.missing_premises) if tree.missing_premises else '',
                        'Step Type': 'forward',
                        'Inputs': " | ".join([tree.get_step_value(x) for x in intermediate.inputs]),
                        'Equation': f"{' + '.join([x for x in intermediate.inputs])} -> i{intermediate_idx}",
                        'Missing Premise Scores': ", ".join(
                            [f'{x:.3f}' for x in intermediate.scores.get('missing_premises', [-1])]
                        ),
                        'Recover Threshold': recover_threshold
                    }

                    new_rows.append(row)

            if 'hypothesis' in write_step_types:
                for hypothesis_idx, hypothesis in enumerate(tree.hypotheses):
                    mp_score = sum(hypothesis.scores.get('missing_premises', [-1])) / max(len(hypothesis.scores.get('missing_premises', [-1])), 1)
                    tree_recovered = True if mp_score >= recover_threshold else tree_recovered

                    row = {
                        'Tree Number': tree_idx,
                        'Tree Recovered': False,
                        'Step Recovered': mp_score >= recover_threshold,
                        'Score': mp_score,
                        'Generation': hypothesis.output,
                        'Missing Premises': ' | '.join(tree.missing_premises) if tree.missing_premises else '',
                        'Step Type': 'abductive',
                        'Inputs': " | ".join([tree.get_step_value(x) for x in hypothesis.inputs]),
                        'Equation': f"{' + '.join([x for x in hypothesis.inputs])} -> h{hypothesis_idx}",
                        'Missing Premise Scores': ", ".join(
                            [f'{x:.3f}' for x in hypothesis.scores.get('missing_premises', [-1])]
                        ),
                        'Recover Threshold': recover_threshold
                    }

                    new_rows.append(row)

            if tree_recovered:
                for idx, row in enumerate(new_rows):
                    new_rows[idx]['Tree Recovered'] = True


            rows.extend(new_rows)

        if not silent:
            print(f"Writing out {len(rows)} for the missing premise csv to {output_file}")

        writer.writeheader()
        writer.writerows(rows)


def recover_premise_csv(
        input_file: Path,
        output_file: Path,
        write_step_types: List[str],
        force_output: bool,
        recover_threshold: float = 0.8,
        silent: bool = False
):
    """
    Wrapper that can export a list of trees from a file or a list of trees where each tree is a list of proofs for that
    tree.

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param write_step_types: For each row in the csv, write out these types of step generations
    :param force_output: Overwrite anything currently written in the output file path.
    :param recover_threshold: How high of a score does the step need before it's "recovered" the premise.
    :param silent: No log messages
    """
    # VALIDATE ARGS
    assert input_file.is_file() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert (not output_file.exists() or force_output) and str(output_file).endswith('.csv'), \
        'Please specify an empty csv file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    if str(input_file).endswith('.jsonl'):
        json_trees = list(jsonlines.open(str(input_file), 'r'))
    else:
        json_trees = json.load(input_file.open('r'))

    if len(json_trees) == 0:
        return

    trees = [Tree.from_json(t) for t in json_trees]

    convert_tree_to_missing_premise_csv(
        output_file,
        trees,
        write_step_types,
        recover_threshold=recover_threshold,
        silent=silent
    )


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over',
                           default=SEARCH_DATA_FOLDER / 'moral40_full.json')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--write_step_types', '-s', type=str, choices=['intermediate', 'hypothesis'], nargs='+',
                           default=['intermediate', 'hypothesis'],
                           help='For each row in the csv, write out these types of step generations')
    argparser.add_argument('--recover_threshold', '-r', type=float, required=True,
                           help='How high of a score does the step need before it\'s "recovered" the premise.')
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _write_step_types: List[str] = args.write_step_types
    _recover_threshold: float = args.recover_threshold
    _force_output: bool = args.force_output

    recover_premise_csv(
        input_file=_input_file,
        output_file=_output_file,
        write_step_types=_write_step_types,
        force_output=_force_output,
        recover_threshold=_recovery_threshold
    )
