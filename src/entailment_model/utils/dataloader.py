from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from jsonlines import jsonlines
from sklearn.model_selection import train_test_split
import random

random.seed(42)
np.random.seed(42)

from entailment_model.utils.eval_config import eval_config
from entailment_model.utils.eval_logger import log


def load_csv(path: str) -> pd.DataFrame:
    """
    Given a path load up a Pandas dataframe

    :param path: String path pointing to the csv file you want to download
    """

    return pd.read_csv(path)


def clean_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, run operations to clean it for proper use
    This should really only contain functions that remove annotator comments or idiosyncrasies (not reduce to 1 label
    etc. since other functions).

    :return: Cleaned dataframe
    """

    # TODO - this may be all the cleaning needed... if so, change the name of this func.
    # We put ? on some of our annotations, this removes them.
    data = data.replace({r'\?': ''}, regex=True)

    return data


def reduce_labels(data: pd.DataFrame, label_columns: List[str]) -> pd.DataFrame:
    """
    Given a dataframe and list of labeled columns, reduce the labeled columns per row to the maximum label occurance
    and store that label in a new column called 'label'

    :param data: Dataframe with multiple labels per row
    :param label_columns: List of columns that contain labels you wish to reduce into 1
    :return: Dataframe with a new column 'label' containing the highest occuring label for each row
    """

    def label_picker(row):
        # Fancy way of getting the max # of items [N, N, E] -> N :)
        row_set = set(row)

        # Remove nan values, we want the max across labeled columns
        if np.nan in row_set:
            row_set.remove(np.nan)

        # If none of the columns were labeled, return -1
        if len(row_set) == 0:
            return -1

        return max(row_set, key=row.count)

    labels = data[label_columns].values.tolist()
    data['label'] = [label_picker(row) for row in labels]

    return data


def label_to_class_index(
        label: str,
        class_idx_table: Dict[str, int] = eval_config.label_to_idx,
        allow_contradictions: bool = eval_config.allow_contradiction_label
):
    """
    Helper function for getting the index of a label given a label and optionally the lookup table to use along with
    a flag for determining whether Contradictions are an allowed label

    :param label: The label you want the class index of
    :param class_idx_table: Table that maps string labels to class indices
    :param allow_contradictions: Boolean toggle for allowing the C (contradiction) label and class index
    :return: An integer representing the class idx of the label
    """

    if not allow_contradictions:
        class_idx_table['C'] = class_idx_table['N']
    return class_idx_table.get(label, -1)


def export_dataset(
        data: pd.DataFrame,
        label_columns: List[str],
        train_export_path: Path = eval_config.dataset_training_file,
        validation_export_path: Path = eval_config.dataset_validation_file,
        allow_contradiction_label: bool = eval_config.allow_contradiction_label,
        max_row: Optional[int] = eval_config.max_row,
        only_annotated_rows: bool = eval_config.only_annotated_rows,
        validation_split: float = eval_config.validation_split,
        use_all_rows_for_validation: bool = False,
        only_use_original_rows: bool = False,
        fold: bool = eval_config.fold
):
    """
    A helper function that will produce 2 dataset jsonl files for training and validation.

    :param data: Dataframe you want to export into jsonl dataset files
    :param label_columns: List of strings indicating the columns that hold labels
    :param train_export_path: The path for the training dataset file
    :param validation_export_path: The path for the validation dataset file
    :param allow_contradiction_label: Allow contradictions in the labels or convert them to N
    :param max_row: Integer representing the maximum row of the dataframe to export
    :param validation_split: Float representing percentage of training rows that will be held out for validation
    :param use_all_rows_for_validation: If true, any annotated row could be used in the heldout set, otherwise use only
        the original 150 rows that were annotated by 3 labelers
    :param only_use_original_rows: Use only the first 150 rows for training and validation
    :param fold: Boolean toggle for creating multiple folds of the dataset where the validation file has a different
        unique set of heldout rows per fold.
    """

    log.info("Exporting dataset")

    # Various ways to trim the data down.
    if max_row:
        data = data[0:max_row]
    if only_annotated_rows:
        data = data.dropna(0, inplace=False, how='all', subset=label_columns)
    if only_use_original_rows:
        data = data[data['idx'].isin(eval_config.originally_annotated_row_indices)]

    data = clean_labels(data)
    data = reduce_labels(data, label_columns)

    premises = data['Output']
    hypothesis = data['Hypothesis']
    labels = data['label']
    indices = data['idx']

    if use_all_rows_for_validation:

        # The lines of json that will be in the file we create for the dataset
        lines_of_json = [
            {
                'premise': x[0],
                'hypothesis': x[1],
                'label': label_to_class_index(x[2], allow_contradictions=allow_contradiction_label)
            }
            for x in zip(premises, hypothesis, labels)
        ]

        unique_hypothesis = np.array(list(sorted(list(set([x['hypothesis'] for x in lines_of_json])))))
        np.random.shuffle(unique_hypothesis)
        unique_hypothesis = unique_hypothesis.tolist()

        additional_training_lines = []

    else:
        # If we only want to use the original 150 rows for validation, we need to keep the unique hypothesis for
        # validation to those rows.

        # grab the unique hypothesis of the 150 rows
        validation_data = data[data['idx'].isin(eval_config.originally_annotated_row_indices)]


        unique_hypothesis = np.array(list(sorted(list(set(validation_data['Hypothesis'].values.tolist())))))
        np.random.shuffle(unique_hypothesis)
        unique_hypothesis = unique_hypothesis.tolist()

        # The lines of json that will be in the file we create for the dataset
        # Drop all hypothesis that are in the unique_hypothesis set but not within the original 150 rows annotated
        # (duplicate hypothesis rows that we annotated later)
        lines_of_json = [
            {
                'premise': x[0],
                'hypothesis': x[1],
                'label': label_to_class_index(x[2], allow_contradictions=allow_contradiction_label)
            }
            for x in zip(premises, hypothesis, labels, indices) if (x[1] in unique_hypothesis and x[3] in eval_config.originally_annotated_row_indices) or x[1] not in unique_hypothesis
        ]

        additional_training_lines = [
            x for x in hypothesis if x not in unique_hypothesis
        ]

    if validation_split > 0:
        if fold:
            num_of_folds = len(np.arange(0, 1, validation_split).tolist())

            split_percentage = validation_split
            old_validation_folds = []
            remaining_lines = unique_hypothesis
            fold_idx = 1
            for fold in range(num_of_folds):
                if split_percentage >= 1.0:
                    break

                train, validation = train_test_split(remaining_lines, test_size=split_percentage, shuffle=True, random_state=42)

                train_fold_path = train_export_path / str(fold_idx)
                train_fold_path.mkdir(parents=True, exist_ok=True)

                validation_fold_path = validation_export_path / str(fold_idx)
                validation_fold_path.mkdir(parents=True, exist_ok=True)

                with jsonlines.open(str(train_fold_path / "train.jsonl"), 'w') as f:
                    training_lines = train + old_validation_folds + additional_training_lines
                    f.write_all([x for x in lines_of_json if x['hypothesis'] in training_lines])
                with jsonlines.open(str(validation_fold_path / "validation.jsonl"), 'w') as f:
                    f.write_all([x for x in lines_of_json if x['hypothesis'] in validation])

                old_validation_folds.extend(validation)
                remaining_lines = train
                split_percentage = len(validation) / len(remaining_lines)
                fold_idx += 1

            train_fold_path = train_export_path / str(fold_idx)
            train_fold_path.mkdir(parents=True, exist_ok=True)

            validation_fold_path = validation_export_path / str(fold_idx)
            validation_fold_path.mkdir(parents=True, exist_ok=True)

            with jsonlines.open(str(train_fold_path / "train.jsonl"), 'w') as f:
                f.write_all([x for x in lines_of_json if x['hypothesis'] in old_validation_folds])
            with jsonlines.open(str(validation_fold_path / "validation.jsonl"), 'w') as f:
                f.write_all([x for x in lines_of_json if x['hypothesis'] in remaining_lines])

        else:
            train_export_path.parent.mkdir(parents=True, exist_ok=True)
            validation_export_path.parent.mkdir(parents=True, exist_ok=True)

            train, validation = train_test_split(unique_hypothesis, test_size=validation_split, shuffle=True, random_state=42)

            with jsonlines.open(str(train_export_path), 'w') as f:
                f.write_all([x for x in lines_of_json if x['hypothesis'] in train])
            with jsonlines.open(str(validation_export_path), 'w') as f:
                f.write_all([x for x in lines_of_json if x['hypothesis'] in validation])

            log.info(f"Training dataset exported to {train_export_path}")
            log.info(f"Validation dataset exported to {validation_export_path}")
    else:
        train_export_path.parent.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(str(train_export_path), 'w') as f:
            f.write_all(lines_of_json)
        log.info(f"All data used for training dataset and exported to {train_export_path}")


if __name__ == "__main__":
    data = load_csv(eval_config.csv_data_path)
    export_dataset(
        data,
        label_columns=eval_config.csv_label_columns,
        train_export_path=eval_config.dataset_training_file,
        validation_export_path=eval_config.dataset_validation_file
    )
