from datetime import datetime
from typing import List, Dict, Union, Any, Optional
from pathlib import Path

from entailment_model.utils.evaluator_harness import run
from entailment_model.utils.eval_config import eval_config
from entailment_model.utils.eval_logger import log
from entailment_model.utils.dataloader import load_csv, export_dataset


class RunConfig:
    """
    This class is used to hold the hyperparameters of a specific run.
    """

    lr: float

    epochs: int   # Number of epochs to train on (ignored if training is set to false)
    train: bool  # Whether or not to train the model prior to evaluation
    huggingface_model: str  # The name of the model to use

    training_dataset_path: Path
    validation_dataset_path: Path
    checkpoints_path: Path
    eval_out_path: Path

    run_name: Optional[str]

    train_on_folds: bool
    __creation_date__: str = str(datetime.now().timestamp()).replace(".", "")  # for unique names

    other_args: List[str]

    def __init__(
            self,
            lr: float = 0.00009,
            epochs: int = 1,
            train: bool = True,
            huggingface_model: str = eval_config.base_evaluation_model,
            training_dataset_path: Path = eval_config.dataset_training_file,
            validation_dataset_path: Path = eval_config.dataset_validation_file,
            checkpoints_path: Path = eval_config.checkpoints_folder,
            eval_out_path: Path = eval_config.eval_out_folder,
            run_name: Optional[str] = None,
            other_args: List[str] = None,
    ):
        self.lr = lr

        self.epochs = epochs
        self.train = train
        self.huggingface_model = huggingface_model

        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path

        self.run_name = run_name

        self.other_args = other_args

        if not self.run_name:
            formatted_model_name = self.huggingface_model.replace('/', '_').replace('-', '_')

            self.run_name = f'{formatted_model_name}_E{self.epochs}_{"Trained" if self.train else "NotTrained"}_{self.__creation_date__}'

        self.model_path = checkpoints_path
        self.eval_out_path = eval_out_path

        folded_path = self.training_dataset_path / '1'
        if folded_path.is_dir():
            self.train_on_folds = True

    def __str__(self):
        return self.run_name

    @property
    def train_args(self) -> List[Union[str, Any]]:
        """
        Get the arguments required for training this configuration of parameters.
        """
        training_args = [
            '--do_train',
            '--task', 'nli',
            '--learning_rate', str(self.lr),
            '--dataset', str(self.training_dataset_path),
            '--output_dir', str(self.model_path),
            '--model', self.huggingface_model,
            '--num_train_epochs', f'{self.epochs}'
        ]

        training_args.extend(self.other_args)

        return training_args

    @property
    def eval_args(self) -> List[Union[str, Any]]:
        """
        Get the arguments required for evaluating this configuration of parameters.
        """

        args = [
            '--do_eval',
            '--task', 'nli',
            '--dataset', str(self.validation_dataset_path),
            '--output_dir', str(self.eval_out_path),
        ]

        # If this run was trained, then there is a folder in the model_path with the weights etc.
        if self.train:
            args.extend(['--model', str(self.model_path)])
        else:
            # If the model was not trained, then we need to use the name of the model from hugging face.
            args.extend(['--model', str(self.huggingface_model)])

        args.extend(self.other_args)

        return args


def runner(
        configs: List[RunConfig],
):
    """
    Run a list of RunConfigs one after another

    :param configs: List of RunConfigs to run
    """

    log.info(f"Beginning a batch of {len(configs)} runs")

    for config in configs:

        log.info(f"Running {str(config)}")

        # Train (if we wanted to train)
        if config.train:
            run(config.train_args)

        # Evaluate
        run(config.eval_args)


    log.info("Runs finished")


if __name__ == "__main__":
    if not eval_config.dataset_training_file.is_file():
        log.info("Creating the dataset.")

        data = load_csv(eval_config.csv_data_path)
        export_dataset(
            data,
            eval_config.csv_label_columns,
            train_export_path=eval_config.dataset_training_file,
            validation_export_path=eval_config.dataset_validation_file
        )

    # Create 1 run with no training, then 5 runs with different numbers of epochs.
    base = RunConfig(train=False)

    run_configs = []
    #run_configs = [RunConfig(epochs=x) for x in range(1, 6)]
    run_configs = [RunConfig(epochs=2)]
    #run_configs.append(base)

    runner(run_configs)
