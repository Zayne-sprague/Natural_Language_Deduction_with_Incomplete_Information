from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
from typing import List
import torch
from multiprocessing import Pool

from utils.paths import DATA_FOLDER
from search.tree.tree import Tree
from search.tree.step_generation import StepGeneration
from search.evaluation.metrics.rouge_metric import RougeEvaluation
from search.evaluation.metrics.entailement_metric import EntailmentEvaluation
from search.evaluation.metrics.rouge_entailment_hm_metric import RougeEntailmentHMMetric
from search.evaluation.metrics.evaluation_metric import EvaluationMetric


def get_scorer(scorer_method: str, torch_device: str):
    if scorer_method == 'rouge':
        return RougeEvaluation()
    if scorer_method == "entailment":
        return EntailmentEvaluation("wanli_entailment_model", torch_device=torch.device(torch_device))
    if scorer_method == "rouge+entailment":
        rscorer = RougeEvaluation()
        escorer = EntailmentEvaluation("wanli_entailment_model", torch_device=torch.device(torch_device))
        return RougeEntailmentHMMetric(rscorer, escorer)


def score_missing_premises(
        steps: List[StepGeneration],
        missing_premises: List[str],
        scorer: EvaluationMetric,
        eval_method: str
):
    predictions = [x.output for x in steps]

    if eval_method == "gold_to_prediction":
        return [scorer.score(predictions, [x]*len(predictions)) for x in missing_premises]
    elif eval_method == "prediction_to_gold":
        return [scorer.score([x]*len(predictions), predictions) for x in missing_premises]
    elif eval_method == "average":
        missing_premise_scores = []
        p2g = [scorer.score(predictions, [x] * len(predictions)) for x in missing_premises]
        g2p = [scorer.score([x] * len(predictions), predictions) for x in missing_premises]

        for idx in range(len(missing_premises)):
            missing_premise_scores.append([(x + y) / 2 for x, y in zip(p2g[idx], g2p[idx])])
        return missing_premise_scores

    elif eval_method == "max":
        missing_premise_scores = []
        p2g = [scorer.score(predictions, [x] * len(predictions)) for x in missing_premises]
        g2p = [scorer.score([x] * len(predictions), predictions) for x in missing_premises]

        for idx in range(len(missing_premises)):
            missing_premise_scores.append([max(x, y) for x, y in zip(p2g[idx], g2p[idx])])
        return missing_premise_scores

    elif eval_method == "min":
        missing_premise_scores = []
        p2g = [scorer.score(predictions, [x] * len(predictions)) for x in missing_premises]
        g2p = [scorer.score([x] * len(predictions), predictions) for x in missing_premises]

        for idx in range(len(missing_premises)):
            missing_premise_scores.append([min(x, y) for x, y in zip(p2g[idx], g2p[idx])])
        return missing_premise_scores
    else:
        raise Exception(f"Invalid eval_method specified: {eval_method}")


def score_goal(
        steps: List[StepGeneration],
        goal: str,
        scorer: EvaluationMetric,
        eval_method: str
):
    predictions = [x.output for x in steps]

    if eval_method == "gold_to_prediction":
        scores = scorer.score([goal] * len(predictions), predictions)
    elif eval_method == "prediction_to_gold":
        scores = scorer.score(predictions, [goal] * len(predictions))
    elif eval_method == "average":
        g2p = scorer.score([goal] * len(predictions), predictions)
        p2g = scorer.score(predictions, [goal] * len(predictions))
        scores = [(x + y) / 2 for x, y in zip(g2p, p2g)]
    elif eval_method == "max":
        g2p = scorer.score([goal] * len(predictions), predictions)
        p2g = scorer.score(predictions, [goal] * len(predictions))
        scores = [max(x, y) for x, y in zip(g2p, p2g)]
    elif eval_method == "min":
        g2p = scorer.score([goal] * len(predictions), predictions)
        p2g = scorer.score(predictions, [goal] * len(predictions))
        scores = [min(x, y) for x, y in zip(g2p, p2g)]
    else:
        raise Exception(f"Invalid eval_method specified: {eval_method}")

    return scores


def score_intermediate_to_hypotheses(
        steps: List[StepGeneration],
        hypotheses: List[StepGeneration],
        scorer: EvaluationMetric,
        eval_method: str
):
    predictions = [x.output for x in steps]

    for idx, hypothesis in enumerate(hypotheses):
        goal = hypothesis.output

        if eval_method == "gold_to_prediction":
            scores = scorer.score([goal]*len(predictions), predictions)
        elif eval_method == "prediction_to_gold":
            scores = scorer.score(predictions, [goal]*len(predictions))
        elif eval_method == "average":
            g2p = scorer.score([goal] * len(predictions), predictions)
            p2g = scorer.score(predictions, [goal]*len(predictions))
            scores = [(x + y) / 2 for x, y in zip(g2p, p2g)]
        elif eval_method == "max":
            g2p = scorer.score([goal] * len(predictions), predictions)
            p2g = scorer.score(predictions, [goal] * len(predictions))
            scores = [max(x, y) for x, y in zip(g2p, p2g)]
        elif eval_method == "min":
            g2p = scorer.score([goal] * len(predictions), predictions)
            p2g = scorer.score(predictions, [goal] * len(predictions))
            scores = [min(x, y) for x, y in zip(g2p, p2g)]
        else:
            raise Exception(f"Invalid eval_method specified: {eval_method}")

        for score, step in zip(scores, steps):
            step_scores = step.scores.get('intermediate_to_hypotheses', {})
            step_scores[f'h{idx}'] = score


def __process_wrapper__(kwargs):
    return __score_searches__(**kwargs)


def score_searches(
        input_file: Path,
        output_file: Path,
        force_output: bool,
        score_method: List[str],
        score_type: List[str],
        eval_methods: List[str],
        score_steps: List[str],
        torch_devices: List[str],
        silent: bool = False,
):
    """
    Wrapper function for __score_searches__

    Specifically, this wrapper helps spread the search across multiple devices by splitting the number of trees to
    search over.

    This splitting is done via multiprocessing and putting each instance of the search on a new core. (might not work if
    you have more devices than cpu cores... but lucky you if that's the case :D )

    :param input_file: Path to file with trees to search over
    :param output_file: Path to the output file that the evaluations will be stored in
    :param force_output: Overwrite anything currently written in the output file path.
    :param score_method: What type of metric to use for the score of each score_type (these values should align with the
        score_types you gave)
    :param eval_methods: How to handle the direction of arguments when scoring a step (separate with space, should have
        1 value for each score_method)
    :param score_type: What to score and save in the output
    :param score_steps: What step types should be scored (intermediates, hypotheses, etc.)
    :param torch_devices: List of devices to split the search across
    :param silent: No log messages
    :return: List of Tree objects where each step is scored according to the given parameters
    """

    # VALIDATE ARGS
    assert input_file.is_file() and str(input_file).endswith('.json'), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'
    assert len(score_method) == len(score_type) == len(eval_methods) and len(score_type) > 0, \
        'Please make sure you have the same number of score_methods, eval_methods, and score_types and you have at' \
        ' least 1 of each'


    # LOAD UP TREES
    json_trees = json.load(input_file.open('r'))
    all_trees = [Tree.from_json(t) for t in json_trees]

    device_trees = []

    trees_per_device = len(all_trees) / len(torch_devices)
    for i in range(len(torch_devices)):
        start_idx = int(min(i * trees_per_device, len(all_trees) - 1))
        end_idx = int(min((i + 1) * trees_per_device, len(all_trees)))

        if i == len(torch_devices) - 1:
            # If its the last device, always take all the trees up till the last index.
            end_idx = len(all_trees)

        device_trees.append(all_trees[start_idx:end_idx])

    score_searches = []

    for idx, (torch_device, trees) in enumerate(zip(torch_devices, device_trees)):

        score_args = {
            'trees': trees,
            'score_method': score_method,
            'score_type': score_type,
            'eval_methods': eval_methods,
            'score_steps': score_steps,
            'torch_device': torch_device,
            'job_idx': idx
        }

        score_searches.append(score_args)

    if len(torch_devices) == 1:
        results = [__score_searches__(**score_searches[0])]
    else:
        with Pool(len(torch_devices)) as p:
            results = p.map(__process_wrapper__, score_searches)

    searched_trees = []
    for result in results:
        searched_trees.extend(result)

    # EXPORT EVALUATIONS.
    with output_file.open('w') as f:
        json.dump([x.to_json() for x in searched_trees], f)

    if not silent:
        print("Finished scoring")


def __score_searches__(
        trees: List[Tree],
        score_method: List[str],
        score_type: List[str],
        eval_methods: List[str],
        score_steps: List[str],
        torch_device: str,
        job_idx: int = 0
) -> List[Tree]:
    """
    Given a file with a list of Trees that have been searched over (expanded intermediates and hypotheses) score those
    expanded steps with respect to the goal, missing premise, etc.

    :param trees: List of tree objects to score
    :param score_method: What type of metric to use for the score of each score_type (these values should align with the
        score_types you gave)
    :param eval_methods: How to handle the direction of arguments when scoring a step (separate with space, should have
        1 value for each score_method)
    :param score_type: What to score and save in the output
    :param score_steps: What step types should be scored (intermediates, hypotheses, etc.)
    :param torch_device: What torch device to use
    :param job_idx: if the search is a part of a multiprocessed job, this is the job index.  This helps position the
        progress bars etc.
    :return: List of Tree objects where each step is scored according to the given parameters
    """

    score_configs = zip(score_type, score_method, eval_methods)

    scorers = {score_name:
                    {'scorer': get_scorer(scorer_name, torch_device=torch_device), 'eval_method': eval_method}
               for score_name, scorer_name, eval_method in score_configs}

    for tree in tqdm(
            trees,
            total=len(trees),
            desc=f"Scoring Trees on device {torch_device}",
            leave=False,
            position=job_idx
    ):

        steps: List[StepGeneration] = []

        if 'intermediates' in score_steps:
            steps.extend(tree.intermediates)
        if 'hypotheses' in score_steps:
            steps.extend(tree.hypotheses)

        if len(steps) == 0:
            continue

        if 'missing_premises' in score_type:
            scores = score_missing_premises(
                steps,
                tree.missing_premises,
                scorers['missing_premises']['scorer'],
                scorers['missing_premises']['eval_method']
            )

            for midx, missing_premise_score in enumerate(scores):
                for idx, score in enumerate(missing_premise_score):
                    step_scores = steps[idx].scores.get('missing_premises', [])
                    step_scores.append(score)
                    steps[idx].scores['missing_premises'] = step_scores

        if 'goal' in score_type:
            scores = score_goal(
                steps,
                tree.goal,
                scorers['goal']['scorer'],
                scorers['goal']['eval_method']
            )

            for idx, score in enumerate(scores):
                steps[idx].scores['goal'] = score

        if 'intermediate_to_hypotheses' in score_type:
            score_intermediate_to_hypotheses(
                steps[0:len(tree.intermediates)],
                tree.hypotheses,
                scorers['intermediate_to_hypotheses']['scorer'],
                scorers['intermediate_to_hypotheses']['eval_method']
            )

    return trees


if __name__ == "__main__":

    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over')
    argparser.add_argument('--output_file', '-o', type=str, required=True,
                           help='Path to the output file that the evaluations will be stored in')
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--score_method', '-sm', type=str, nargs="+",
                           choices=["rouge", "entailment", "rouge+entailment"],
                           help="What type of metric to use for the score of each score_type (these values should "
                                "align with the score_types you gave)")
    argparser.add_argument('--score_type', '-st', type=str, nargs="+",
                           choices=["goal", "missing_premises", "valid_step", "intermediate_to_hypotheses"],
                           help='What to score and save in the output (separate with space, should have 1 value for '
                                'each score_method)')
    argparser.add_argument('--eval_methods', '-mpem', type=str, nargs='+',
                           choices=["gold_to_prediction", "prediction_to_gold", "average", "max", "min"],
                           help='How to handle the direction of arguments when scoring a step (separate with space, '
                                'should have 1 value for each score_method)')
    argparser.add_argument('--score_steps', '-ss', type=str, nargs="+", choices=["intermediates", "hypotheses"],
                           help='What step types should be scored (intermediates, hypotheses, etc.)')
    argparser.add_argument('--torch_devices', '-td', type=str, nargs='+', default=['cpu'],
                           help='Torch devices to use for the inner search, if multiple devices given the trees will be'
                                ' split across them then merged back into a single output file.')

    args = argparser.parse_args()

    _input_file: Path = Path(args.input_file)
    _output_file: Path = Path(args.output_file)
    _force_output: bool = args.force_output
    _score_method: List[str] = args.score_method
    _score_type: List[str] = args.score_type
    _eval_methods: List[str] = args.eval_methods
    _score_steps: List[str] = args.score_steps
    _torch_devices: List[str] = args.torch_devices

    score_searches(
        input_file=_input_file,
        output_file=_output_file,
        force_output=_force_output,
        score_method=_score_method,
        eval_methods=_eval_methods,
        score_type=_score_type,
        torch_devices=_torch_devices,
        score_steps=_score_steps
    )
