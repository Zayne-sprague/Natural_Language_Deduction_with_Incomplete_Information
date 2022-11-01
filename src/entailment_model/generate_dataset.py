import argparse
from pathlib import Path
import shutil
import os
from typing import List

from entailment_model.utils.argparser import add_dataset_arguments
from entailment_model.utils.dataloader import load_csv, export_dataset
from entailment_model.utils.paths import ENT_OUT_DIR, ENT_DATASET_DIR, ENT_CHECKPOINTS_DIR
from entailment_model.utils.eval_config import eval_config


def generate_dataset(
        dataset_name: str = 'tmp_dataset',
        validation_percentage: float = 0.33,
        folded: bool = True,
        label_columns: List[str] = eval_config.csv_label_columns,
        use_all_rows_for_validation: bool = False,
        only_use_original_rows: bool = False,
):
    """
    This function will create a dataset folder that is recognised by the training and evaluation scripts at the location
    specified.  It also holds various parameters for creating datasets with unique properties.

    :param dataset_name: Name of the dataset you want to create (default value will have the risk of being overwritten)
    :param validation_percentage: The percentage of how many unique hypothesis should be held out for validation
    :param folded: Whether or not to do cross-fold validation where the # of folds is determined by the
        validation_percentage
    :param label_columns: Which columns to use as annotations for the dataset
    :param use_all_rows_for_validation: Use all the unique hypothesis that have annotations for evaluation, if false
        use only the original 150 rows for evaluation
    :param only_use_original_rows: Use only the original 150 rows for training and validation
    """

    dataset_path: Path = ENT_DATASET_DIR / dataset_name

    # Do not overwrite a folder if something already exists there UNLESS the folder is the tmp_dataset
    if dataset_path.exists():
        if dataset_name == 'tmp_dataset':
            if dataset_path.is_dir():
                shutil.rmtree(str(dataset_path))
            elif dataset_path.is_file():
                os.remove(str(dataset_path))
        else:
            raise Exception(f"A folder with this name already exists, {dataset_path}")

    data = load_csv(eval_config.csv_data_path)

    # Produce the training and validation dataset files.
    export_dataset(
        data,
        label_columns=label_columns,
        train_export_path=dataset_path if folded else dataset_path / 'train.jsonl',
        validation_export_path=dataset_path if folded else dataset_path / 'validation.jsonl',
        validation_split=validation_percentage,
        only_annotated_rows=True,
        use_all_rows_for_validation=use_all_rows_for_validation,
        only_use_original_rows=only_use_original_rows,
        fold=folded
    )

    # Export every row (including unlabeled) this is for inference and calculating entropy
    export_dataset(
        data,
        label_columns=label_columns,
        train_export_path=dataset_path / 'all.jsonl',
        validation_split=0.,
        only_annotated_rows=False,
        max_row=None,
        only_use_original_rows=only_use_original_rows,
        fold=False
    )


if __name__ == "__main__":
    """For when you call this script directly"""

    parser = argparse.ArgumentParser()

    add_dataset_arguments(parser)

    args = parser.parse_args()

    _dataset_name = args.dataset_name
    _validation_percentage = args.validation_percentage
    _folded = args.folded
    _label_columns = args.label_columns
    _use_all_rows_for_validation = args.use_all_rows_for_validation
    _only_use_original_rows = args.only_use_original_rows

    generate_dataset(
        dataset_name=_dataset_name,
        validation_percentage=_validation_percentage,
        folded=_folded,
        label_columns=_label_columns,
        use_all_rows_for_validation=_use_all_rows_for_validation,
        only_use_original_rows=_only_use_original_rows
    )
