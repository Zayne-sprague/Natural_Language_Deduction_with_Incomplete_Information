from entailment_model.utils.paths import ROOT_PROJECT_DIR, ENT_DATASET_DIR, ENT_CHECKPOINTS_DIR, ENT_OUT_DIR

from pathlib import Path
from typing import List, Dict
import os


class EvalConfig:
    """
    Helpful configuration for running the evaluation scripts on our manually annotated entailment labels.
    """

    # TODO - allows you to use a google sheets doc for annotations -- turned this off for the public release.
    # Information about where to find the Google Spreadsheet for downloading/parsing
    csv_sheet_id: str = None
    csv_sheet_name: str = None  # sheet to parse
    csv_data_path: str = str(ROOT_PROJECT_DIR / 'annotated_entailment_data.csv')

    # List of column names that hold labels (must match whats in the sheet)
    csv_label_columns: List[str] = [
        '3',
        '2',
        '1'
    ]

    # Slice the training dataframe to the maximum number of rows
    max_row: int = None
    # Only allow rows that have at least 1 annotation column populated
    only_annotated_rows: bool = True

    # Originally annotated rows by 3 labelers (our original benchmarks were ran on these rows)
    originally_annotated_row_indices: List[int] = list(range(2, 152))

    # Where to export the dataset
    dataset_training_file: Path = ENT_DATASET_DIR / 'more_annotations/p2/training'
    dataset_validation_file: Path = ENT_DATASET_DIR / 'more_annotations/p2/validation'

    # Directory that will hold all the models checkpoints by default
    checkpoints_folder = ENT_CHECKPOINTS_DIR / 'more_annotations/p2'
    checkpoints_folder.mkdir(parents=True, exist_ok=True)

    eval_out_folder = ENT_OUT_DIR / 'more_annotations/p2'
    eval_out_folder.mkdir(parents=True, exist_ok=True)

    # Label to Class IDX
    label_to_idx: Dict[str, int] = {
        'E': 2,  # entailment
        'N': 1,  # neutral
        'C': 0,  # contradiction
    }

    # Class IDX to Label
    idx_to_label: Dict[str, int] = {
        2: 'E',  # entailment
        1: 'N',  # neutral
        0: 'C',  # contradiction
    }

    # If this is false, all functions will default to converting C into N unless specified otherwise.
    allow_contradiction_label: bool = False

    # validation split, percentage (0.33 == 33%) of the training data will be held out for validation
    validation_split: float = 0.333
    fold: bool = True

    # Threshold the model has to predict above in order to be considered entailment
    entailment_threshold: float = 0.2

    # Model that functions will default too when evaluating
    base_evaluation_model: str = 'microsoft/deberta-base-mnli'

    # GPUs we can use ( 0 indexed and comma seperated)
    allowed_gpu_idxs: str = "0,1"


eval_config = EvalConfig()

# Restrict Pytorch and HuggingFace to the GPUs we selected
os.environ["CUDA_VISIBLE_DEVICES"] = eval_config.allowed_gpu_idxs
