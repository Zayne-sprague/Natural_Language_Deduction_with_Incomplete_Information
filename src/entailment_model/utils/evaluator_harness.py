import sys
import json

from entailment_model.utils.eval_logger import log
from entailment_model.utils.paths import ENT_CHECKPOINTS_DIR, FP_DATASET_ARTIFACTS_DIR

try:
    if str(FP_DATASET_ARTIFACTS_DIR) not in sys.path:
        sys.path.append(str(FP_DATASET_ARTIFACTS_DIR))
    from entailment_model.fp_dataset_artifacts.run import main
except ImportError as e:
    log.critical(e)
    log.critical("Please run ./install.sh in the evaluation folder before using functions in this file!")
    sys.exit(1)

from typing import List, Any, Union
from unittest.mock import patch


def run(cmd_args: List[Union[str, Any]]):
    """
    Will run the main function in fp-dataset-artifacts with a given set of command-line arguments

    To avoid changing code that was written for the fp-dataset-artifacts repo, we add arguments to the argv variable
    directly.  Gross, I know, but it also means we don't have to chagne the other repo... which is nice... kinda?
    The patch() function will remove the added argv params on ext.

    :param cmd_args: List of command line arguments you want the run.py function to grab
    """

    # Argv expect that the first argument is the python file currently running.  To avoid remembering this idiosyncrasy
    # we add it manually here.
    cmd_args.insert(0, 'run.py')

    # Create a patch that will overwrite the argv parameters for every call within the context
    with patch('sys.argv', cmd_args):
        # Run the cool code :)
        main()


if __name__ == "__main__":
    from entailment_model.utils.eval_config import eval_config

    model_file = ENT_CHECKPOINTS_DIR / 'test_model'
    model_file.mkdir(parents=True, exist_ok=True)

    # Train
    args = [
        '--do_eval',
        '--task', 'nli',
        '--dataset', str(eval_config.dataset_training_file),
        '--output_dir', str(model_file),
        '--model', 'microsoft/deberta-base-mnli',
    ]
    run(args)

    #
    # # Train
    # args = [
    #     '--do_train',
    #     '--task', 'nli',
    #     '--dataset', str(eval_config.dataset_training_file),
    #     '--output_dir', str(model_file),
    #     '--model', 'microsoft/deberta-base-mnli',
    #     '--num_train_epochs', '1'
    # ]
    #
    # run(args)
    #
    # # Evaluate
    # args = [
    #     '--do_eval',
    #     '--task', 'nli',
    #     '--dataset', str(eval_config.dataset_validation_file),
    #     '--output_dir', str(model_file),
    #     '--model', str(model_file),
    #     '--num_train_epochs', '1'
    # ]
    #
    # run(args)
