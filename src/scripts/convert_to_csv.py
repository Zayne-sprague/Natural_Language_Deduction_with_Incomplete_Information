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


def convert_tree_to_csv(
        output_file: Path,
        trees: List[Tree],
        write_step_types: List[str],
        silent: bool = False
):

    CSV_FIELDNAMES = [
        'Tree Number',
        'Goal',
        'Missing Premises',
        'Generation',
        'Step Type',
        'Inputs',
        'Equation',
        'Goal Score',
        'Missing Premise Scores'
    ]

    with output_file.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        rows = []
        for tree_idx, tree in enumerate(trees):
            if 'intermediate' in write_step_types:
                for intermediate_idx, intermediate in enumerate(tree.intermediates):
                    row = {
                        'Tree Number': tree_idx,
                        'Goal': tree.goal,
                        'Missing Premises': ' | '.join(tree.missing_premises) if tree.missing_premises else '',
                        'Generation': intermediate.output,
                        'Step Type': 'forward',
                        'Inputs': " | ".join([tree.get_step_value(x) for x in intermediate.inputs]),
                        'Equation': f"{' + '.join([x for x in intermediate.inputs])} -> i{intermediate_idx}",
                        'Goal Score': f"{intermediate.scores.get('goal', -1):.3f}",
                        'Missing Premise Scores': ", ".join(
                            [f'{x:.3f}' for x in intermediate.scores.get('missing_premises', [-1])]
                        )
                    }

                    rows.append(row)

            if 'hypothesis' in write_step_types:
                for hypothesis_idx, hypothesis in enumerate(tree.hypotheses):
                    row = {
                        'Tree Number': tree_idx,
                        'Goal': tree.goal,
                        'Missing Premises': ' | '.join(tree.missing_premises) if tree.missing_premises else '',
                        'Generation': hypothesis.output,
                        'Step Type': 'abductive',
                        'Inputs': " | ".join([tree.get_step_value(x) for x in hypothesis.inputs]),
                        'Equation': f"{' + '.join([x for x in hypothesis.inputs])} -> i{hypothesis_idx}",
                        'Goal Score': f"{hypothesis.scores.get('goal', -1):.3f}",
                        'Missing Premise Scores': ", ".join(
                            [f'{x:.3f}' for x in hypothesis.scores.get('missing_premises', [-1])]
                        )

                    }

                    rows.append(row)

        writer.writeheader()
        writer.writerows(rows)


def convert_proofs_to_csv(
        output_file: Path,
        proofs: List[List[Tree]],
        silent: bool = False
):

    CSV_FIELDNAMES = [
        'Tree Number',
        'Proof Number',
        'Goal',
        'Missing Premises',
        'Step Type',
        'Inputs',
        'Generation',
        'Equation',
        'Goal Score',
        'Missing Premise Scores',
        'GPT3 Step Score',
        'Forward Model Entailment'
    ]

    with output_file.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        proof_count = 0

        rows = []
        for tree_idx, tree in enumerate(proofs):

            for proof_idx, proof in enumerate(tree):
                proof_count += 1

                for intermediate_idx, intermediate in enumerate(proof.intermediates):

                    def premise_id(x):
                        if 'p' not in x:
                            return x

                        idx = int(x[1:])
                        if idx >= len(proof.__original_premises__):
                            return f'p{idx}*'
                        return x

                    row = {
                        'Tree Number': tree_idx,
                        'Proof Number': proof_idx,
                        'Goal': proof.goal,
                        'Missing Premises': ' | '.join(proof.missing_premises) if proof.missing_premises else '',
                        'Generation': intermediate.output,
                        'Step Type': intermediate.tags.get('step_type', 'unk'),
                        'Inputs': " | ".join([proof.get_step_value(x) for x in intermediate.inputs]),
                        'Equation': f"{' + '.join([premise_id(x) for x in intermediate.inputs])} -> i{intermediate_idx}",
                        'Goal Score': f"{intermediate.scores.get('goal', -1):.3f}",
                        'Missing Premise Scores': ", ".join(
                            [f'{x:.3f}' for x in intermediate.scores.get('missing_premises', [-1])]
                        ),
                        'GPT3 Step Score': intermediate.scores.get('gpt3_output_score', -1),
                        'Forward Model Entailment': intermediate.scores.get('forward_agreement', -1)
                    }

                    rows.append(row)

        if not silent:
            print(f"Writing out {len(rows)} rows for {proof_count} proofs from {len(proofs)} trees, saved in"
                  f" {str(output_file)}")
        writer.writeheader()
        writer.writerows(rows)


def convert_to_csv(
        input_file: Path,
        output_file: Path,
        write_step_types: List[str],
        force_output: bool,
        silent: bool = False
):
    """
    Wrapper that can export a list of trees from a file or a list of trees where each tree is a list of proofs for that
    tree.

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param write_step_types: For each row in the csv, write out these types of step generations
    :param force_output: Overwrite anything currently written in the output file path.
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

    is_proofs_file = isinstance(json_trees[0], list) or isinstance(json_trees[0], tuple)

    if is_proofs_file:
        proofs = [[Tree.from_json(t) for t in p] for p in json_trees]

        convert_proofs_to_csv(output_file, proofs, silent=silent)
    else:
        trees = [Tree.from_json(t) for t in json_trees]

        convert_tree_to_csv(output_file, trees, write_step_types, silent=silent)


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over',
                           default=SEARCH_DATA_FOLDER / 'moral40_full.json')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--write_step_types', '-s', type=str, choices=['intermediate', 'hypothesis'], nargs='+',
                           default=['intermediate', 'hypothesis'],
                           help='For each row in the csv, write out these types of step generations')
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _write_step_types: List[str] = args.write_step_types
    _force_output: bool = args.force_output

    convert_to_csv(
        input_file=_input_file,
        output_file=_output_file,
        write_step_types=_write_step_types,
        force_output=_force_output
    )
