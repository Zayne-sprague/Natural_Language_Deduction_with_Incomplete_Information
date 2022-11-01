import argparse
import numpy as np
from pathlib import Path
import shutil
import os

from entailment_model.utils.argparser import add_entropy_arguments
from entailment_model.utils.dataloader import load_csv
from entailment_model.utils.eval_config import eval_config
from entailment_model.utils.results_handler import ResultsHandler
from entailment_model.utils.runner import RunConfig, runner
from entailment_model.utils.evaluation_predictions import EvaluationPredictions
from entailment_model.utils.paths import ENT_OUT_DIR, ENT_CHECKPOINTS_DIR, ENT_DATASET_DIR


def entropy_scores(
        run_name: str = "tmp_model",
        dataset_name: str = "tmp_dataset"
):
    """
    Function that produces entropy scores per row of a dataset.

    :param run_name: Name of the model to use (must be a model that was not folded)
    :param dataset_name: Name of the dataset to use
    """

    model_path: Path = ENT_CHECKPOINTS_DIR / run_name
    if not model_path.is_dir():
        raise Exception(f"The run name path does not have a checkpoint {model_path}")
    if (model_path / '1').is_dir():
        raise Exception(f'Models with folds cannot be ran for Entropy')

    dataset_path = ENT_DATASET_DIR / dataset_name
    data_file = dataset_path / 'all.jsonl'

    if not data_file.exists():
        raise Exception(f'Dataset file could not be found {data_file}')

    eval_out = ENT_OUT_DIR / f'{run_name}_entropy'
    if eval_out.exists():
        if run_name == 'tmp_model':
            if eval_out.is_dir():
                shutil.rmtree(str(eval_out))
            elif eval_out.is_file():
                os.remove(str(eval_out))
        else:
            raise Exception(f"Evaluation on entropy for this run has already been completed {eval_out}")

    run = RunConfig(
        epochs=0,
        train=False,
        validation_dataset_path=data_file,
        eval_out_path=eval_out
    )

    runner([run])

    eval_predictions = EvaluationPredictions.load_from_folder(eval_out)
    results_handler = ResultsHandler()

    annotations = load_csv(eval_config.csv_data_path)
    results_handler.set_manual_annotations(annotations)

    results_handler.register_predictions(eval_predictions, 'predictions')

    ents = results_handler.high_entropy_annotations('predictions')
    entropy_col = np.zeros(len(annotations))
    for ent in ents:
        for idx in ent[0].tolist():
            entropy_col[idx] = ent[1]

    annotations['New Entropy'] = entropy_col

    annotations.to_csv("entropy_out.csv")

    return entropy_col


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_entropy_arguments(parser)

    args = parser.parse_args()

    _run_name = args.run_name
    _dataset_name = args.dataset_name

    entropy_scores(
        run_name=_run_name,
        dataset_name=_dataset_name
    )
