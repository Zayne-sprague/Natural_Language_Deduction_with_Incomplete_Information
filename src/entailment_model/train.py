import argparse
from pathlib import Path
import shutil
import os
from typing import List

from entailment_model.utils.argparser import add_training_arguments
from entailment_model.utils.runner import RunConfig, runner
from entailment_model.utils.paths import ENT_OUT_DIR, ENT_DATASET_DIR, ENT_CHECKPOINTS_DIR
from entailment_model.utils.eval_config import eval_config


def train(
    dataset_name: str = "tmp_dataset",
    run_name: str = "tmp_model",
    epochs: int = 2,
    lr: float = 0.00005,
    huggingface_model: str = eval_config.base_evaluation_model,
    other_args: List[str] = ()
):
    """
    :param dataset_name: Name of the dataset to use
    :param run_name: Name of the model (default value has the risk of being overwritten later)
    :param epochs: How many epochs to train for
    :param lr: Learning rate of the model
    :param huggingface_model: The uri of the Huggingface Model to use.
    :param other_args: Other arguments to pass to the HFTrainer that were not explicitly set
    """

    dataset_path = ENT_DATASET_DIR / dataset_name
    eval_out_path = ENT_OUT_DIR / run_name
    checkpoint_path = ENT_CHECKPOINTS_DIR / run_name
    should_train = epochs > 0

    runs = []

    # Do not overwrite a folder if something already exists there UNLESS the folder is the tmp_model
    if eval_out_path.exists() or checkpoint_path.exists():
        if run_name == 'tmp_model':
            if eval_out_path.is_dir():
                shutil.rmtree(str(eval_out_path))
            elif eval_out_path.is_file():
                os.remove(str(eval_out_path))

            if checkpoint_path.is_dir():
                shutil.rmtree(str(checkpoint_path))
            elif checkpoint_path.is_file():
                os.remove(str(checkpoint_path))

        else:
            raise Exception(f"A folder already exists with the name {run_name} at these paths {eval_out_path} or "
                            f"{checkpoint_path}")

    folded_path = ENT_DATASET_DIR / dataset_name / '1'
    if folded_path.is_dir():
        # Folded
        if not should_train:
            raise Exception("For folded datasets, you have to specify a number of epochs greater than 0.")

        for fold in dataset_path.glob("*"):
            if not fold.is_dir():
                continue

            runs.append(
                RunConfig(
                    lr=lr,
                    epochs=epochs,
                    training_dataset_path=fold / 'train.jsonl',
                    validation_dataset_path=fold / 'validation.jsonl',
                    checkpoints_path=checkpoint_path / fold.name,
                    eval_out_path=eval_out_path / fold.name,
                    huggingface_model=huggingface_model,
                    run_name=run_name,
                    other_args=other_args
                )
            )

    else:
        # Not folded
        runs.append(

            RunConfig(
                lr=lr,
                epochs=epochs,
                train=should_train,
                training_dataset_path=dataset_path / 'train.jsonl',
                validation_dataset_path=dataset_path / 'validation.jsonl' if should_train else dataset_path / 'train.jsonl',
                checkpoints_path=checkpoint_path,
                eval_out_path=eval_out_path,
                huggingface_model=huggingface_model,
                run_name=run_name,
                other_args=other_args
            )
        )

    runner(runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_training_arguments(parser)

    args, unknown = parser.parse_known_args()

    _epochs = args.epochs
    _lr = args.learning_rate
    _run_name = args.run_name
    _huggingface_model = args.huggingface_model
    _dataset_name = args.dataset_name

    train(
        dataset_name=_dataset_name,
        run_name=_run_name,
        epochs=_epochs,
        lr=_lr,
        huggingface_model=_huggingface_model,
        other_args=unknown,
    )
