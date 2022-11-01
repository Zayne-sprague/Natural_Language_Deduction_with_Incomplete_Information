import argparse
from pathlib import Path
import json
from typing import Dict, List
import os

from entailment_model.train import train
from entailment_model.evaluate import evaluate
from entailment_model.utils.eval_config import eval_config
from entailment_model.utils.eval_logger import log
from entailment_model.utils.paths import EVAL_OUTS_DIR
from entailment_model.utils.argparser import add_grid_search_arguments


def read_out_file(f: Path) -> Dict:
    """Read in the eval out file"""
    if not f.exists():
        return {}
    return json.load(f.open('r'))


def write_out_file(f: Path, data: Dict):
    """Write to the eval file"""
    return json.dump(data, f.open('w'))


def u(updates, data=None):
    """Helper for doing nested dict updates without overwriting stuff, inspired by updeep in js."""

    if data is None:
        data = {}
    if not isinstance(updates, dict):
        return updates

    for k, v in updates.items():
        data[k] = u(v, data.get(k))

    return data


def grid_search(
        findings_file_name: str = "tmp_grid_search_findings.json",
        dataset_name: str = "tmp_dataset",
        run_name: str = "tmp_model",
        epochs: List[int] = (2,),
        lrs: List[float] = (0.00005,),
        huggingface_models: List[str] = (eval_config.base_evaluation_model,)
):
    findings_file_path: Path = EVAL_OUTS_DIR / findings_file_name

    if findings_file_path.exists():
        if findings_file_name == 'tmp_grid_search_findings':
            if eval_out_path.is_file():
                os.remove(str(eval_out_path))
        else:
            raise Exception(f"A folder already exists with the name {run_name} at these paths {eval_out_path} or "
                            f"{checkpoint_path}")

    for huggingface_model in huggingface_models:
        for epoch_num in epochs:
            for lr in lrs:

                evaluations = read_out_file(findings_file_path)

                train(
                    dataset_name=dataset_name,
                    run_name=run_name,
                    epochs=epoch_num,
                    lr=lr,
                    huggingface_model=huggingface_model
                )

                _, _, best_scores, _ = evaluate(
                    run_name=run_name
                )

                updated_evaluations = u(
                    {huggingface_model:
                         {str(epoch_num):
                              {str(lr):
                                   {'score': best_scores['f1'], 'thresh': best_scores['t']}
                               }
                          }
                     }, evaluations
                )

                write_out_file(findings_file_path, updated_evaluations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_grid_search_arguments(parser)

    args = parser.parse_args()

    _dataset_name = args.dataset_name
    _run_name = args.run_name

    _epochs = args.epochs
    _lrs = args.learning_rates
    _huggingface_models = args.huggingface_models
    _findings_file_name = args.findings_file_name

    grid_search(
        findings_file_name=_findings_file_name,
        dataset_name=_dataset_name,
        run_name=_run_name,
        epochs=_epochs,
        lrs=_lrs,
        huggingface_models=_huggingface_models,
    )
