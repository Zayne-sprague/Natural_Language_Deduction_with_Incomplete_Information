import nltk

# from evaluation.utils.dataloader import load_csv, clean_labels, reduce_labels, label_to_class_index

import pandas as pd
from typing import List, Dict
from nltk.metrics.agreement import AnnotationTask
from itertools import cycle
from textwrap import dedent
from sklearn.metrics import f1_score


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
    data = data.replace({r'\*': ''}, regex=True)
    data = data.replace('NR', 'YR')
    data = data.replace('NL', 'YL')
    data = data.replace('R', 'N')
    data = data.replace('(bad inputs)', '')
    data = data.replace('Vacuous', 'N')

    return data


# def labeler_f1(data: pd.DataFrame, label_columns: List[str]) -> Dict[str, float]:
#     """
#     Given a dataframe with label columns, find the F1 score of each label column vs consensus.
#     :param data: Dataframe that contains label columns we want to evaluate over
#     :param label_columns: The specific columns we want to both create ground truth (majority vote) labels and create f1
#         scores over
#     :return: Dictionary where the keys are the label column names and the values are their respective f1 scores.
#     """
#
#     data = data[:eval_config.max_row]
#     data = clean_labels(data)
#     data = reduce_labels(data, label_columns)
#
#     labels = [label_to_class_index(x) for x in data['label'].values]
#
#     f1_scores = {}
#     for annotator in label_columns:
#         preds = [label_to_class_index(x) for x in data[annotator].values]
#
#         f1 = f1_score(labels, preds, pos_label=eval_config.label_to_idx['E'], zero_division=0)
#         f1_scores[annotator] = f1
#
#     return f1_scores


def annotation_agreement(data: pd.DataFrame, label_columns: List[str]) -> nltk.AnnotationTask:
    """
    Create metrics on the annotation agreement between the labeled columns in a dataframe.
    :param data: Dataframe that has all the information
    :param label_columns: Columns in the dataframe that have the labels
    :return: AnnotationTask with metrics on the annotations in the dataframe.
    """

    print("Running NLTKs annotator agreement.")

    labels = data[label_columns]
    formatted_labels = []
    for label in label_columns:
        # Format for NLTK AnnotationTask is [(annotator, row_index, label), (...), ...]
        formatted_labels.extend(list(zip(cycle([label]), list(labels[label].index), labels[label].tolist())))

    metrics = AnnotationTask(formatted_labels)

    return metrics


def log_annotation_metrics(metrics: nltk.AnnotationTask):
    """
    Logs various statistics in the AnnotationTask metrics
    :param metrics: An NLTK AnnotationTask that has various statistics of interest to log
    """

    pair_wise_aggr_logs = []
    pair_wise_kappa_logs = []

    labelers = set(metrics.C.copy())
    others = labelers.copy()

    for labeler in labelers:
        others.remove(labeler)

        for other in others:
            pair_wise_aggr_logs.append(
                f'Agreement between {labeler} & {other} : {metrics.Ao(labeler, other)}'
            )
            pair_wise_kappa_logs.append(
                f'Kappa between {labeler} & {other} Kappa: {metrics.kappa_pairwise(labeler, other)}, Ae Kappa: {metrics.Ae_kappa(labeler, other)}'
            )

    pair_wise_aggr_msg = "\n\t\t\t".join(pair_wise_aggr_logs)
    pair_wise_kappa_msg = "\n\t\t\t".join(pair_wise_kappa_logs)

    print(dedent(f"""
        --- Annotation Metrics --
        Average Agreement: {metrics.avg_Ao()}
        \t{pair_wise_aggr_msg}
        ---
        Kappa: {metrics.kappa()}
        Multi Kappa: {metrics.multi_kappa()}
        \t{pair_wise_kappa_msg}
        ---
        Pi: {metrics.pi()}
        S: {metrics.S()}
        --- ---
        """
    ))


if __name__ == "__main__":
    data = pd.read_csv('../3b_step_valid_r.csv')
    labels = ['Valid? (Zayne)', 'Valid? (Kaj)', 'Valid? (Greg)']

    data = clean_labels(data)

    metrics = annotation_agreement(data, labels)

    # metrics = annotation_agreement(clean_labels(load_csv(eval_config.csv_data_path)), eval_config.csv_label_columns)
    log_annotation_metrics(metrics)


    data = pd.read_csv('../3b_step_valid_ur.csv')
    labels = ['Valid? (Zayne)', 'Valid? (Kaj)', 'Valid? (Greg)']

    data = clean_labels(data)

    metrics = annotation_agreement(data, labels)

    # metrics = annotation_agreement(clean_labels(load_csv(eval_config.csv_data_path)), eval_config.csv_label_columns)
    log_annotation_metrics(metrics)

    # data = load_csv(eval_config.csv_data_path)
    # scores = labeler_f1(data, eval_config.csv_label_columns)
    #
    # for labeler, f1 in scores.items():
    #     print(f'{labeler} f1 score: {f1:.2f}')