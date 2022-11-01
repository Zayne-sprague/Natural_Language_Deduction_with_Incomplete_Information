from pathlib import Path
import uuid
from copy import deepcopy
from typing import List, Dict, Optional
from jsonlines import jsonlines
import numpy as np
import math

from entailment_model.utils.eval_config import eval_config
from entailment_model.utils.runner import RunConfig


class EvaluationPredictions:
    # Short cuts into various values in the predictions array
    predicted_labels: List[int] = []
    gold_labels: Dict[str, int] = dict()
    entailment_scores: List[float] = []
    scores: List[List[float]] = []
    entropy: List[float] = []

    # Current set of predictions, these can change and update depending on the entailment threshold.
    __predictions__: List[Dict[str, any]]

    # The entailment threshold
    __entailment_threshold__: float

    # What is considered to be the positive label for our metrics (default is the Entailment label)
    pos_label: int

    __id__ = uuid.uuid4()
    __fold_num__ = Optional[int]

    def __init__(
            self,
            predictions: List[Dict[str, any]],
            entailment_threshold: float = eval_config.entailment_threshold,
            pos_label: int = eval_config.label_to_idx['E'],
    ):
        self.__predictions__ = predictions
        self.entailment_threshold = entailment_threshold
        self.pos_label = pos_label

        # Despite changing entailment thresholds, these should never change.
        # Not every row or prediction could have a gold label, especially if this is a test set run.  So since it could
        # be sparse, gold labels is a lookup table not an array.
        self.gold_labels = {
                idx: x['label'] for idx, x in enumerate(self.predictions) if x.get('label',-1) > -1
        }

        # Despite changing entailment thresholds, these should never change.
        E = eval_config.label_to_idx['E']
        self.scores = [__softmax__(x['predicted_scores']).tolist() for x in self.predictions]
        self.entailment_scores = [x[E] for x in self.scores]

        self.entropy = self.__compute_prediction_entropy__()

    @classmethod
    def load_from_folder(cls, path: Path, **kwargs):
        """
        helper function to create an EvaluationPrediction class given a folder location of a models output.
        If the output was folded, i.e. multiple models on different unique subsets were run, it will merge the
        predictions into one class instance and return it.

        :param path: The path to the models output OR the top level of the folded path
        """

        # TODO - the check for if a folder is a folded run is lazy, think of better ways
        folded_path = path / '1'
        if folded_path.is_dir():
            preds = []
            for pth in path.glob("*"):
                if pth.is_dir():
                    # Get all the predictions for this fold and add it to our global list
                    preds.extend(__read_predictions__(pth))
        else:
            preds = __read_predictions__(path)

        # Create the EvaluationPredictions class
        eval_pred = cls(
            predictions=preds,
            **kwargs
        )
        return eval_pred

    @classmethod
    def load_from_run_config(cls, runner: RunConfig):
        """
        Helper function for creating an EvaluationPrediction class instance given a RunConfig.
        """

        pth = runner.eval_out_path
        return cls.load_from_folder(pth)

    def __getitem__(self, item):
        """
        Allows the use of

            eval_pred = EvaluationPredictions(...)
            prediction_at_idx = eval_pred[idx]

        and returns the prediction for that index.
        """
        return self.predictions[item]

    @property
    def entailment_threshold(self) -> float:
        return self.__entailment_threshold__

    @entailment_threshold.setter
    def entailment_threshold(self, threshold: float):
        """
        Changing the entailment threshold will change the predictions of the model as well
        """

        self.__entailment_threshold__ = threshold
        self.__threshold_predictions__()

    @property
    def predictions(self) -> List[Dict[str, any]]:
        return self.__predictions__

    @predictions.setter
    def predictions(self, new_predictions):
        """
        Helper for updating the new predictions but also recalculating the predicted_labels property
        """
        self.__predictions__ = deepcopy(new_predictions)

        C = eval_config.label_to_idx['C']
        N = eval_config.label_to_idx['N']
        self.predicted_labels = [x['predicted_label'] if x['predicted_label'] != C else N for x in new_predictions]

    def __threshold_predictions__(self):
        """
        Iterate over the entailment scores from the model and if the score is above the entailment threshold, predict
        entailment, otherwise predict neutral.
        """

        new_predictions = deepcopy(self.predictions)

        E = eval_config.label_to_idx['E']
        N = eval_config.label_to_idx['N']

        for idx, entailment_score in enumerate(self.entailment_scores):
            if entailment_score >= self.entailment_threshold:
                new_predictions[idx]['predicted_label'] = E
            else:
                # If entailment is below the threshold, but still higher than everything else, still set it to N because
                # it's too low of a score.
                new_predictions[idx]['predicted_label'] = N

        self.predictions = new_predictions

    def __compute_prediction_entropy__(self):
        """
        Entropy function for binary classification.  Range is (0, 1) where 1 is high in entropy and needs more labeling!
        """

        E = eval_config.label_to_idx["E"]
        N = eval_config.label_to_idx["N"]
        C = eval_config.label_to_idx["C"]

        e_n_scores = [[x[E], x[N] + x[C]] for x in self.scores]

        entropy = []
        for score in e_n_scores:
            ent = score[0] * math.log2(score[0] + 1e-10) + score[1] * math.log2(score[1] + 1e-10)
            norm = math.log2(2)
            entropy.append(- ent / norm)
        return entropy


def __softmax__(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def __read_predictions__(path: Path) -> List[Dict[str, any]]:
    """
    Read in the eval_predictions file from the results folder
    """

    predictions_file = path / 'eval_predictions.jsonl'

    if not predictions_file.is_file():
        # If there are no predictions, most of this class is useless and shouldn't have been called.
        raise Exception(f"No predictions file was found at {predictions_file}")

    with jsonlines.open(str(predictions_file), mode='r') as f:
        preds = []
        for line in f:
            preds.append(line)

        return preds
