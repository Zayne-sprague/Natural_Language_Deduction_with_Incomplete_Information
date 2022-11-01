from typing import List, Dict
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import re

from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration


def evaluate_tree(
        tree: Tree,
        thresholds: List[float],
        positive_labels: List[str],
        confusion_matrix: Dict[str, Dict[str, any]] = None,
        filter_non_labeled: bool = False
):
    best_threshold = -1
    last_threshold = -1
    best_is_last = False

    if not confusion_matrix:
        confusion_matrix = {str(t): {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for t in thresholds}

    intermediates = tree.intermediates
    not_last_intermediates = intermediates[:-1]
    not_last_intermediates.extend(tree.hypotheses)

    for iidx, intermediate in enumerate(not_last_intermediates):  # type: StepGeneration
        score = intermediate.scores.get('goal', -1)
        labels = intermediate.annotations.get('goal', [])

        if filter_non_labeled and len(labels) == 0:
            continue

        label = False
        if any([x in labels for x in positive_labels]):
            label = True

        for t in thresholds:
            if score >= t and label:
                confusion_matrix[str(t)]['tp'] += 1
            elif score < t and label:
                confusion_matrix[str(t)]['fn'] += 1
            elif score >= t and not label:
                confusion_matrix[str(t)]['fp'] += 1
            elif score < t and not label:
                confusion_matrix[str(t)]['tn'] += 1

            if score >= t > best_threshold:
                best_threshold = t

    if len(tree.intermediates) > 0:
        last_score = tree.intermediates[-1].scores.get('goal', -1)

        labels = tree.intermediates[-1].annotations.get('goal', [])

        label = False
        if any([x in labels for x in positive_labels]):
            label = True

        for t in thresholds:
            if last_score >= t and label:
                confusion_matrix[str(t)]['tp'] += 1
            elif last_score < t and label:
                confusion_matrix[str(t)]['fn'] += 1
            elif last_score >= t and not label:
                confusion_matrix[str(t)]['fp'] += 1
            elif last_score < t and not label:
                confusion_matrix[str(t)]['tn'] += 1

            if t < last_score:
                last_threshold = t
                break

    return best_threshold, last_threshold, best_is_last, confusion_matrix


def cm_precision(confusion_matrix: Dict[str, Dict[str, any]], threshold: float):
    cm = confusion_matrix[str(threshold)]
    d = (cm['tp'] + cm['fp'])
    if d == 0:
        return 0
    return cm['tp'] / d


def cm_recall(confusion_matrix: Dict[str, Dict[str, any]], threshold: float):
    cm = confusion_matrix[str(threshold)]

    d = (cm['tp'] + cm['fn'])
    if d == 0:
        return 0
    return cm['tp'] / d


def cm_f1(confusion_matrix: Dict[str, Dict[str, any]], threshold: float):
    p = cm_precision(confusion_matrix, threshold)
    r = cm_recall(confusion_matrix, threshold)

    d = p + r
    if d == 0:
        return 0
    return 2 * (p * r) / d


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over', required=True)
    argparser.add_argument('--thresholds', '-t', type=float, nargs="+", default=[],
                           help='Specific thresholds to evaluate on')
    argparser.add_argument('--threshold_range_step', '-trs', type=float,
                           help='All values between 0 and 1 will be tested with a step size of this value, these will'
                                'be added to the specific thresholds if any were set. (turned off by default)')
    argparser.add_argument('--silent', '-s', action='store_true', dest='silent', help='do not print metrics')
    argparser.add_argument('--output_file', '-o', type=str, help='json file to store the evaluation results, if not'
                                                                 'specified -- nothing will be saved.')
    argparser.add_argument('--force', '-f', action='store_true', dest='force', help='overwrite output_file if exists')
    argparser.add_argument('--metrics', '-m', type=str, nargs='+',
                           choices=['precision', 'recall', 'f1', 'goals_entailed', 'last_intermediate_entailed'],
                           default=['goals_entailed', 'last_intermediate_entailed'],
                           help="What evaluation metric would you like to track.  'goals_entailed' and "
                                "'last_intermediate_entailed' uniquely do not require on annotations to be present, all"
                                " other metrics require an annotation label on the step generation.")
    argparser.add_argument('--positive_labels', '-pl', type=str, nargs='+',
                           help="When calculating metrics like precision, these values determine which class labels are"
                                "associated with good step generations (i.e. -pl 2 3, would mean any labels on step "
                                "generations with a value or 2 or 3 will be treated as a positive step generation)")

    args = argparser.parse_args()

    input_file: Path = Path(args.input_file)
    specific_thresholds: List[float] = args.thresholds
    threshold_step: float = args.threshold_range_step
    silent: bool = args.silent
    output_file: Path = Path(args.output_file) if args.output_file else None
    force: bool = args.force
    report_metrics: List[str] = args.metrics
    positive_labels: List[str] = args.positive_labels

    # For f-string, how many 0s should be padded to make everything consistent with same padding.
    threshold_zero_pad_len: int = 2

    thresholds = specific_thresholds
    if threshold_step:
        thresholds.extend([x for x in np.arange(0, 1, threshold_step).tolist() if x not in thresholds])

        zero_pad = len(re.search('\d+\.(0*)', str(threshold_step)).group(1)) + 1
        threshold_zero_pad_len = zero_pad if zero_pad > threshold_zero_pad_len else threshold_zero_pad_len

    for t in specific_thresholds:
        zero_pad = len(re.search('\d+\.(0*)', str(t)).group(1))
        threshold_zero_pad_len = zero_pad if zero_pad > threshold_zero_pad_len else threshold_zero_pad_len

    # VALIDATE ARGS
    assert input_file.is_file() and str(input_file).endswith('.json'), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file or not output_file.exists() or force, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    json_trees = json.load(input_file.open('r'))
    trees = [Tree.from_json(t) for t in json_trees]

    goals_entailed = {str(t): {
        'goals_entailed': 0,
        'last_intermediate': 0,
    } for t in thresholds}
    last_intermediate_is_best = 0

    sorted_thresholds = list(sorted(thresholds, reverse=True))

    confusion_matrix = {str(t): {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0} for t in sorted_thresholds}

    for tree in tqdm(trees, desc='Evaluating Trees', total=len(trees)):

        best_threshold, last_threshold, best_is_last, confusion_matrix = evaluate_tree(
            tree,
            sorted_thresholds,
            positive_labels,
            confusion_matrix,
            filter_non_labeled=len(set(report_metrics) - {'goals_entailed', 'last_intermediate_entailed'}) > 0
        )

        last_intermediate_is_best += 1 if best_is_last else 0
        for t in sorted_thresholds:
            if t < best_threshold:
                goals_entailed[str(t)]['goals_entailed'] += 1
            if t < last_threshold:
                goals_entailed[str(t)]['last_intermediate'] += 1

    if not silent:
        print(f"Total Trees: {len(trees)}")

    metrics = {str(t): {} for t in thresholds}

    for t in thresholds:
        if 'goals_entailed' in report_metrics:
            goals_entailed_percent = goals_entailed[str(t)]['goals_entailed'] / len(trees)
            metrics[str(t)]['goals_entailed'] = goals_entailed_percent
        if 'last_intermediate_entailed' in report_metrics:
            last_entailed_percent = goals_entailed[str(t)]['last_intermediate'] / len(trees)
            metrics[str(t)]['last_intermediate_entailed'] = last_entailed_percent
        if 'precision' in report_metrics:
            metrics[str(t)]['precision'] = cm_precision(confusion_matrix, t)
        if 'recall' in report_metrics:
            metrics[str(t)]['recall'] = cm_recall(confusion_matrix, t)
        if 'f1' in report_metrics:
            metrics[str(t)]['f1'] = cm_f1(confusion_matrix, t)

        if not silent:
            print(f'{t:.{threshold_zero_pad_len}f}: {" | ".join([f"{k}: {v:.3f}" for k,v in metrics[str(t)].items()])}')

    if output_file:
        with output_file.open('w') as f:
            json.dump(metrics, f)
