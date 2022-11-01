from argparse import ArgumentParser
from tqdm import tqdm
import json
from pathlib import Path
from typing import List, Dict

from search.tree import Tree


def find_tree_proofs(
        input_file: Path,
        output_file: Path,
        force_output: bool,
        proof_methods: List[str],
        threshold_values: List[float],
        silent: bool = False
) -> List[List[Tree]]:
    """
    Given a file with a list of scored searched Tree objects, find all possible 'proofs' for each tree where a 'proof'
    is a subset of generations that start from the given premises and conclude with a statement that entails the goal
    of the tree.

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param proof_methods: Metric to use for determining if a tree has successfully proven the goal.
    :param threshold_values: Thresholds for the corresponding --proof_methods, give 1 value to share across all proof
        methods or give 1 threshold per proof method given.
    :param silent: No log messages
    :return: A List of Lists of trees where each sub-list is a set of possible proofs for a tree.  A proof contains only
        the step generations required from going from a set of premises to the goal.
    """

    # VALIDATE ARGS
    assert input_file.is_file() and str(input_file).endswith('.json'), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'
    assert len(proof_methods) == len(threshold_values) or len(threshold_values) == 1, \
        'Make sure to either give a threshold per proof method or a single threshold to share across all proof methods.'

    if len(threshold_values) == 1:
        thresholds: Dict[str, float] = {k: threshold_values[0] for k in proof_methods}
    else:
        thresholds: Dict[str, float] = {k: v for k, v in zip(proof_methods, threshold_values)}

    # LOAD UP TREES
    json_trees = json.load(input_file.open('r'))
    trees: List[Tree] = [Tree.from_json(t) for t in json_trees]

    proofs: List[List[Tree]] = []

    solves = []

    for idx, tree in tqdm(enumerate(trees), total=len(trees), desc='Finding proofs'):

        solutions_for_current_tree = []

        for iidx, intermediate in enumerate(tree.intermediates):

            if 'IntermediateToGoal' in proof_methods:
                goal_score = intermediate.scores.get('goal', -1)

                t = thresholds['IntermediateToGoal']

                if goal_score >= t:
                    proof_tree = Tree.from_json(tree.to_json())
                    proof_tree.reduce_to_intermediate_subtree(iidx)

                    solutions_for_current_tree.append(proof_tree)

            if 'IntermediateToHypothesis' in proof_methods:
                t = thresholds['IntermediateToHypothesis']

                hscores = intermediate.scores.get('intermediate_to_hypotheses', {})
                for hidx, hypothesis in enumerate(tree.hypotheses):
                    hscore = hscores.get(f'h{hidx}', -1)

                    if hscore >= t:
                        proof_tree = Tree.from_json(tree.to_json())

                        proof_tree.bridge_intermediate_to_hypotheses(iidx, hidx)

                        solutions_for_current_tree.append(proof_tree)

        for hidx, hypothesis in enumerate(tree.hypotheses):
            if 'RecoveredMissingPremise' in proof_methods:
                goal_scores = hypothesis.scores.get('missing_premises', [-1])
                goal_score = sum(goal_scores) / max(1, len(goal_scores))

                t = thresholds['RecoveredMissingPremise']

                if goal_score >= t:
                    proof_tree = Tree.from_json(tree.to_json())
                    solutions_for_current_tree.append(proof_tree.hypothesis_to_intermediates(hidx))

        if len(solutions_for_current_tree) > 0:
            proofs.append(solutions_for_current_tree)
            solves.append(True)
        else:
            solves.append(False)

    with output_file.open('w') as f:
        json.dump([[y.to_json() for y in x] for x in proofs], f)

    if not silent:
        for idx, solved in enumerate(solves):
            if solved:
                print(f"Tree {idx} Solved")
            else:
                print(f'Tree {idx} not Solved')

        print(f'Found proofs for {len(proofs)} trees. All proofs are saved at {str(output_file)}')


    return proofs


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over')
    argparser.add_argument('--output_file', '-o', type=str, required=True,
                           help='Path to the output file that the evaluations will be stored in')
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--proof_methods', '-pm', type=str, nargs="+",
                           choices=[
                               'IntermediateToHypothesis',
                               'IntermediateToGoal',
                               'RecoveredMissingPremise',
                           ],
                           help="Metric to use for determining if a tree has successfully proven the goal.")
    argparser.add_argument('--thresholds', '-t', type=float, nargs='+',
                           help='Thresholds for the corresponding --proof_methods, give 1 value to share across all'
                                'proof methods or give 1 threshold per proof method given.')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _proof_methods: List[str] = args.proof_methods
    _threshold_values: List[float] = args.thresholds

    find_tree_proofs(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        proof_methods=_proof_methods,
        threshold_values=_threshold_values
    )
