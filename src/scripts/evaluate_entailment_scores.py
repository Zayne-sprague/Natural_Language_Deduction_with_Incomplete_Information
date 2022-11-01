"""
TODO - A lot of this is copied from scripts/score_searches.py, we should try to combine them together so the func is
    shared and generalized.
"""

from search.evaluation import EvaluationMetric, RougeEntailmentHMMetric, EntailmentEvaluation, RougeEvaluation

from pathlib import Path
import torch
from typing import List, Tuple, Dict
import enum
import pandas as pd
from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class EntailmentMethods(enum.Enum):
    pred_to_target = 'pred_to_target'
    target_to_pred = 'target_to_pred'
    mutual = 'mutual'
    max = 'max'
    min = 'min'


def get_scorer(scorer_metric: str, torch_device: str):
    if scorer_metric == "entailment":
        return EntailmentEvaluation("wanli_entailment_model", torch_device=torch.device(torch_device))
    if scorer_metric == "rouge+entailment":
        rscorer = RougeEvaluation()
        escorer = EntailmentEvaluation("wanli_entailment_model", torch_device=torch.device(torch_device))
        return RougeEntailmentHMMetric(rscorer, escorer)


def score_samples(
        predictions: List[str],
        targets: List[str],
        entailment_method: EntailmentMethods,
        evaluation_metric: EvaluationMetric
) -> List[float]:
    if entailment_method == EntailmentMethods.target_to_pred:
        scores = evaluation_metric.score(targets, predictions)
    elif entailment_method == EntailmentMethods.pred_to_target:
        scores = evaluation_metric.score(predictions, targets)
    elif entailment_method == EntailmentMethods.mutual:
        t2p = score_samples(predictions, targets, EntailmentMethods.target_to_pred, evaluation_metric)
        p2t = score_samples(predictions, targets, EntailmentMethods.pred_to_target, evaluation_metric)
        scores = [x + y for x, y in zip(t2p, p2t)]
    elif entailment_method == EntailmentMethods.max:
        t2p = score_samples(predictions, targets, EntailmentMethods.target_to_pred, evaluation_metric)
        p2t = score_samples(predictions, targets, EntailmentMethods.pred_to_target, evaluation_metric)
        scores = [max(x, y) for x, y in zip(t2p, p2t)]
    elif entailment_method == EntailmentMethods.min:
        t2p = score_samples(predictions, targets, EntailmentMethods.target_to_pred, evaluation_metric)
        p2t = score_samples(predictions, targets, EntailmentMethods.pred_to_target, evaluation_metric)
        scores = [min(x, y) for x, y in zip(t2p, p2t)]
    else:
        raise Exception(f"Unknown entailment method: {entailment_method}")

    return scores


def evaluate_entailment_scores(
    input_file: Path,
    torch_device: str,
    silent: bool = False,
):
    """
    :param input_file: Path to file with trees to search over
    :param score_metric: What type of metric to use for the score of each score_type (these values should align with the
        score_types you gave)
    :param eval_methods: How to handle the direction of arguments when scoring a step (separate with space, should have
        1 value for each score_method)
    :param torch_devices: List of devices to split the search across
    :param silent: No log messages
    :return: List of Tree objects where each step is scored according to the given parameters
    """
    # VALIDATE ARGS
    assert input_file.is_file() and str(input_file).endswith('.csv'), \
        'Please specify a correct path to an annotated csv file.'

    data = pd.read_csv(input_file)

    # data = data[data['File Name'] != '/Users/zaynesprague/Research/NLP/deduce/multi_type_search/output/drake2/denali_exp_r3/eb/d2/no_valid_af_10s_small_sample/output/scored_tree_proofs__abductive_and_forward.json']

    preds = data['Generated Premise'].tolist()
    targets = data['Goal Premise'].tolist()

    raw_labels = zip(data['Valid? (Kaj)'].tolist(), data['Valid? (Zayne)'].tolist(), data['Valid? (Greg)'].tolist())
    labels = []

    def is_positive(x):
        if not isinstance(x, str):
            return False

        invalids = ['L']
        invalid = any([y in x for y in invalids])

        valids = ['Y', 'R']
        valid = any([y in x for y in valids])

        if invalid:
            return False
        if valid:
            return True

        return False


    for raw in raw_labels:
        p = [1 if is_positive(x) else 0 for x in raw if isinstance(x, str)]
        y = sum([1 for x in p if x])
        n = sum([1 for x in p if not x])

        if y > n:
            labels.append(True)
        else:
            labels.append(False)

    assert len(labels) == len(targets) == len(preds), \
        'Something went wrong, targets, preds, and labels are not same length.'

    metric = get_scorer(scorer_metric='rouge+entailment', torch_device=torch_device)
    # metric = get_scorer(scorer_metric='entailment', torch_device=torch_device)

    p2t = score_samples(preds, targets, entailment_method=EntailmentMethods.pred_to_target, evaluation_metric=metric)
    t2p = score_samples(preds, targets, entailment_method=EntailmentMethods.target_to_pred, evaluation_metric=metric)

    p2t_scores = deepcopy(p2t)
    t2p_scores = deepcopy(t2p)
    mutual_scores = [(x + y) / 2 for x, y in zip(deepcopy(p2t), deepcopy(t2p))]
    max_scores = [max(x, y) for x, y in zip(deepcopy(p2t), deepcopy(t2p))]

    p2t_eval = eval_scores(p2t_scores, labels)
    t2p_eval = eval_scores(t2p_scores, labels)
    mutual_eval = eval_scores(mutual_scores, labels)
    max_eval = eval_scores(max_scores, labels)

    print_eval(p2t_eval, 'Generation To Target')
    print_eval(t2p_eval, 'Target To Generation')
    print_eval(mutual_eval, 'Mutual Entailment')
    print_eval(max_eval, 'Max Direction')

def eval_scores(scores: List[float], labels: List[bool]):

    thresholds = np.arange(0.0, 1.0, 0.01).tolist()

    threshold_results = {}
    for t in thresholds:
        preds = [1 if x >= t else 0 for x in scores]

        f1 = f1_score(labels, preds)
        p = precision_score(labels, preds)
        r = recall_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        threshold_results[f'{t:.02f}'] = {'f1': f1, 'p': p, 'r': r, 'cm': cm}

    return threshold_results


def print_eval(eval: Dict[str, Dict[str, any]], name: str):

    print(f"---- ---- ---- {name} ---- ---- ----")
    for k, v in eval.items():
        print(f'{k}: f1 = {v["f1"]:.02f} p = {v["p"]:.02f} r = {v["r"]:.02f}')




if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str,
                           help='Path to csv file with predictions, targets, and gold labels')
    argparser.add_argument('--torch_device', '-td', type=str, default='cpu',
                           help='Torch device to use.')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _torch_device: List[str] = args.torch_device

    evaluate_entailment_scores(
        input_file=_input_file,
        torch_device=_torch_device,
    )

# Allow Y
# Current: 0.70: f1 = 0.57 p = 0.39 r = 1.00
# Best Mutual: 0.76: f1 = 0.88 p = 0.94 r = 0.82

# Allow Y | NR
# Current: 0.70: f1 = 0.81 p = 0.68 r = 1.00
# Best target 2 generation: 0.71: f1 = 0.86 p = 0.81 r = 0.93
# Best Mutual: 0.62: f1 = 0.70 p = 0.77 r = 0.64

# Allow Y | NR | NL
# Current 0.70: f1 = 0.88 p = 0.78 r = 1.00
# Best target 2 generation: 0.29: f1 = 0.90 p = 0.81 r = 1.00
# best mutual is in the 30s
