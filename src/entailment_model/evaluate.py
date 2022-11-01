import argparse
import numpy as np
from pathlib import Path
from typing import List

from entailment_model.utils.argparser import add_evaluation_arguments
from entailment_model.utils.dataloader import load_csv
from entailment_model.utils.eval_config import eval_config
from entailment_model.utils.eval_logger import log
from entailment_model.utils.results_handler import ResultsHandler
from entailment_model.utils.evaluation_predictions import EvaluationPredictions
from entailment_model.utils.paths import ENT_OUT_DIR


def get_metrics(res_handler, t, a, r, p, cut_off_for_t: int = 2):
    metrics = res_handler.get_performance_measures('predictions')

    out = f't @ {t:.{cut_off_for_t}f} | f1 = {metrics["f1"]:.2f}'

    if a:
        out += f' a = {metrics["accuracy"]:.2f}'
    if r:
        out += f' r = {metrics["recall"]:.2f}'
    if p:
        out += f' p = {metrics["precision"]:.2f}'

    return metrics, out


def evaluate(
        run_name: str = 'tmp_model',
        specific_thresholds: List[int] = (.47,),
        all_thresholds: bool = False,
        all_thresholds_step: float = 0.01,
        include_precision: bool = False,
        include_recall: bool = False,
        include_accuracy: bool = False,
        export_bad_predictions: bool = False,
        should_log: bool = False,

):
    """
    Function that will evaluate a run according to various specifications.

    :param run_name: Name of the run you want to evaluate
    :param specific_thresholds: Specific thresholds you want to test
    :param all_thresholds: Check a range of thresholds between 0 and 1
    :param all_thresholds_step: The step between the thresholds in the range 0 and 1 (only matters if all_thresholds is
        set to True)
    :param include_precision: Include precision in the log messages (only matters if should_log is True)
    :param include_recall: Include recall in the log messages (only matters if should_log is True)
    :param include_accuracy: Include accuracy in the log messages (only matters if should_log is True)
    :param export_bad_predictions: Whether or not bad predictions should be exported into a csv file and returned
    :param should_log: Whether or not to log messages about the performance of the run
    """

    results_path: Path = ENT_OUT_DIR / run_name
    if not results_path.is_dir():
        raise Exception(f"The run name path does not have evaluation results {results_path}")

    eval_predictions = EvaluationPredictions.load_from_folder(results_path)
    results_handler = ResultsHandler()

    annotations = load_csv(eval_config.csv_data_path)
    results_handler.set_manual_annotations(annotations)

    results_handler.register_predictions(eval_predictions, 'predictions')

    all_threshold_values = []

    if all_thresholds and should_log:
        if should_log:
            log.info("--- All Threshold Results ---")

        thresholds = np.arange(0, 1, all_thresholds_step).tolist()

        # How many characters do we need to show the value of t
        cut_off_for_t = len(str(all_thresholds_step).split(".")[-1])

        for threshold in thresholds:
            eval_predictions.entailment_threshold = threshold
            metrics, out = get_metrics(results_handler, threshold, include_accuracy, include_recall, include_precision,
                                       cut_off_for_t=cut_off_for_t)
            metrics['t'] = threshold
            all_threshold_values.append(metrics)

            if should_log:
                log.info(out)

    specific_threshold_values = []

    if len(specific_thresholds) > 0:
        if should_log:
            log.info("--- Specific Threshold Results ---")

        for threshold in specific_thresholds:
            eval_predictions.entailment_threshold = threshold

            # How many characters do we need to show the value of t
            cut_off_for_t = len(str(threshold).split(".")[-1])
            metrics, out = get_metrics(results_handler, threshold, include_accuracy, include_recall, include_precision,
                          cut_off_for_t=cut_off_for_t)

            metrics['t'] = threshold
            specific_threshold_values.append(metrics)

            if should_log:
                log.info(out)

    if should_log:
        log.info('--- Finding Best Threshold ---')

    thresholds = np.arange(0, 1, 0.001).tolist()
    best_score = 0.
    best_threshold = -1
    all_best_metrics = {}
    for threshold in thresholds:
        eval_predictions.entailment_threshold = threshold
        metrics = results_handler.get_performance_measures('predictions')
        f1 = metrics['f1']
        if f1 > best_score:
            best_score = f1
            best_threshold = threshold
            all_best_metrics = metrics
    best_metrics = {'t': best_threshold, 'f1': best_score}

    log.info(f'{all_best_metrics}')

    if should_log:
        log.info(f't @ {best_threshold:.3f} | f1 = {best_score:.2f}')

    errors_col = np.zeros(len(annotations))
    if export_bad_predictions:
        errors = results_handler.prediction_errors('predictions')
        for err in errors:
            for idx in err:
                errors_col[idx] = 1

        annotations['New Errors'] = errors_col

        annotations.to_csv("bad_predictions_out.csv")

    return all_threshold_values, specific_threshold_values, best_metrics, errors_col


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_evaluation_arguments(parser)

    args = parser.parse_args()

    _run_name = args.run_name
    _specific_thresholds = args.threshold
    _all_thresholds = args.test_all_thresholds
    _all_thresholds_step = args.all_thresholds_step

    _include_accuracy = args.include_accuracy
    _include_recall = args.include_recall
    _include_precision = args.include_precision

    _export_bad_predictions = args.export_bad_predictions

    evaluate(
        run_name=_run_name,
        specific_thresholds=_specific_thresholds,
        all_thresholds=_all_thresholds,
        all_thresholds_step=_all_thresholds_step,
        include_precision=_include_precision,
        include_recall=_include_recall,
        include_accuracy=_include_accuracy,
        export_bad_predictions=_export_bad_predictions,
        should_log=True
    )
