from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import jsonlines
import json
from copy import deepcopy
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, confusion_matrix

from entailment_model.utils.eval_logger import log
from entailment_model.utils.eval_config import eval_config
from entailment_model.utils.evaluation_predictions import EvaluationPredictions


class ResultsHandler:

    annotated_labels: List[int]
    # What is considered to be the positive label for our metrics (default is the Entailment label)
    pos_label: int = eval_config.label_to_idx['E']

    # Annotations, if given, and the mapping between the prediction index and the annotation index.
    annotations: Optional[pd.DataFrame]
    __annotation_map__: Dict[str, Dict[int, List[int]]] = dict()

    __evaluation_predictions__: Dict[str, EvaluationPredictions] = dict()

    def register_predictions(self, predictions: EvaluationPredictions, name: str = None):
        if not name:
            name = predictions.__id__
        self.__evaluation_predictions__[name] = predictions
        self.__build_annotation_map__()

    def remove_predictions(self, predictions: EvaluationPredictions, name: str = None):
        if not name:
            name = predictions.__id__
        if name in self.__evaluation_predictions__:
            del self.__evaluation_predictions__[predictions.__id__]
        self.__build_annotation_map__()

    @staticmethod
    def __get_preds_and_golds__(predictions: EvaluationPredictions):
        gold_label_indices = list(predictions.gold_labels.keys())
        gold_labels = list(predictions.gold_labels.values())

        C = eval_config.label_to_idx['C']
        N = eval_config.label_to_idx['N']

        gold_labels = [x if x != C else N for x in gold_labels]
        preds = [predictions.predicted_labels[x] if predictions.predicted_labels[x] != C else N for x in gold_label_indices]

        return preds, gold_labels, gold_label_indices

    def set_manual_annotations(self, annotations: pd.DataFrame):
        """
        Helper function for setting manual annotations

        This function will match each row of the annotations dataframe to a prediction and then update the
        annotation_map with the key being the prediction idx and the value being the dataframe idx.  This allows you
        to look up the manual annotations for the prediction through the map without storing everything together
        """

        # Reset the annotation properties
        self.annotations = annotations
        self.__build_annotation_map__()

    def __build_annotation_map__(self):
        self.__annotation_map__ = dict()

        for predictions_key in self.__evaluation_predictions__:
            predictions = self.__evaluation_predictions__[predictions_key]

            self.__annotation_map__[predictions_key] = dict()

            for idx, prediction in enumerate(predictions):
                # Find the rows in the annotations dataframe that has the same premise and hypothesis as the current
                # prediction
                data_row = self.annotations[
                    (self.annotations['Output'] == prediction['premise'])
                    &
                    (self.annotations['Hypothesis'] == prediction['hypothesis'])
                    ]

                if len(data_row) < 1:
                    # If none match, ignore this row
                    continue

                self.__annotation_map__[predictions_key][idx] = data_row.index.values

    def get_performance_measures(self, name: str):
        predictions = self.__evaluation_predictions__.get(name)
        if not predictions:
            return -1

        preds, golds, _ = self.__get_preds_and_golds__(predictions)

        f1 = f1_score(golds, preds, pos_label=self.pos_label, zero_division=0)
        accuracy = accuracy_score(golds, preds)
        precision = precision_score(golds, preds, pos_label=self.pos_label, zero_division=0)
        recall = recall_score(golds, preds, pos_label=self.pos_label, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(golds, preds, labels=[self.pos_label, eval_config.label_to_idx['N']]).ravel()

        return {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'confusion_matrix': [tp, fp, fn, tp]}

    def __get_bridging_scores_v_prediction_score__(self, name: str):

        predictions = self.__evaluation_predictions__.get(name)
        annotation_map = self.__annotation_map__.get(name)

        if not predictions or not annotation_map:
            return [], []

        bridge_scores = []
        entailment_scores = []

        for pred_idx, annotation_idx in annotation_map.items():
            bridge_scores.append(self.annotations.iloc[annotation_idx[0]]['Bridging score'])
            entailment_scores.append(predictions.entailment_scores[pred_idx])

        return bridge_scores, entailment_scores

    def get_bridging_score_mse(self, name: str):
        """
        Helper function for calculating the Mean Squared Error between the annotation files Bridging Score and the
        models predicted score for entailment.
        """

        bridging_scores, entailment_scores = self.__get_bridging_scores_v_prediction_score__(name)
        if len(bridging_scores) == 0 or len(entailment_scores) == 0:
            return -1

        return mean_squared_error(bridge_scores, entailment_scores)

    def high_entropy_annotations(self, name: str):
        predictions = self.__evaluation_predictions__.get(name)
        annotation_map = self.__annotation_map__.get(name)

        if not predictions or not annotation_map:
            return []

        entropy_and_map = []
        for pred_idx, annotation_idx in annotation_map.items():
            entropy_and_map.append([annotation_idx, predictions.entropy[pred_idx]])

        return list(sorted(entropy_and_map, key=lambda x: x[1], reverse=True))

    def prediction_errors(self, name: str):

        predictions = self.__evaluation_predictions__.get(name)
        preds, golds, indices = self.__get_preds_and_golds__(predictions)
        annotation_map = self.__annotation_map__.get(name)

        if len(preds) == 0 or len(golds) == 0 or not annotation_map:
            return []

        errors = []
        for idx, pred_idx in enumerate(indices):
            if preds[idx] != golds[idx]:
                errors.append(annotation_map[pred_idx])

        return errors



if __name__ == "__main__":
    from entailment_model.utils.paths import ENT_OUT_DIR
    from entailment_model.utils.dataloader import load_csv

    annotations = load_csv(eval_config.csv_data_path)
    print(len(annotations))

    res = ResultsHandler()
    res.set_manual_annotations(annotations)

    eval_preds = EvaluationPredictions.load_from_folder(ENT_OUT_DIR / 'more_annotations/p2' / 'microsoft_deberta_base_mnli_E2_Trained_1640983263614248') #EvaluationPredictions.load_from_folder(EVAL_OUT_DIR / 'folded_testing')
    all_eval_preds = EvaluationPredictions.load_from_folder(ENT_OUT_DIR / 'more_annotations/p1_all' / 'microsoft_deberta_base_mnli_E2_Trained_1640981342768023') #EvaluationPredictions.load_from_folder(EVAL_OUT_DIR / 'folded_testing')

    orig_eval_preds = EvaluationPredictions.load_from_folder(ENT_OUT_DIR / 'folded_test_1' / 'microsoft_deberta_base_mnli_E2_Trained_1640063414650421')

    res.register_predictions(eval_preds, 'test')
    res.register_predictions(all_eval_preds, 'all')
    res.register_predictions(orig_eval_preds, 'orig')

    #print(res.get_performance_measures('test'))

    eval_preds.entailment_threshold = 0.32
    #print(res.get_performance_measures('test'))

    #
    ents = res.high_entropy_annotations('all')
    entropy_col = np.zeros(len(annotations))
    for ent in ents:
        for idx in ent[0].tolist():
            entropy_col[idx] = ent[1]
        rows = (ent[0]+2).tolist()
        rows = [str(x) for x in rows]
        #print(f'{", ".join(rows)} @ {str(ent[1])}')


    annotations['new_entropy'] = entropy_col.tolist()

    errors = res.prediction_errors('test')
    errors_col = np.zeros(len(annotations))
    for err in errors:
        for idx in err:
            errors_col[idx] = 1
        rows = err + 2
        rows = [str(x) for x in rows]
        #print(f'{", ".join(rows)}')

    annotations['new_bad_prediction'] = errors_col

    #annotations.to_csv("out.csv")
    thresholds = np.arange(0, 1, 0.01).tolist()

    for t in thresholds:
        eval_preds.entailment_threshold = t
        orig_eval_preds.entailment_threshold = t
        
        mets = res.get_performance_measures('test')
        orig_mets = res.get_performance_measures('orig')
        f1 = mets['f1']
        rec = mets['recall']
        per = mets['precision']

        of1 = orig_mets['f1']
        orec = orig_mets['recall']
        oper = orig_mets['precision']

        print(f"{t:.2f}: f1 = {f1:.2f} rec = {rec:.2f} per = {per:.2f} ||| f1 = {of1:.2f} rec = {orec:.2f} per = {oper:.2f}")


    

