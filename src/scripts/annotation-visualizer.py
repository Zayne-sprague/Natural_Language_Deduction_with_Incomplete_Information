from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict
import json
from jsonlines import jsonlines
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt

from utils.paths import SEARCH_DATA_FOLDER
from search.tree.tree import Tree


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_file', '-i', type=str, help='Path to file with trees to search over')
    argparser.add_argument('--annotation_type', '-at', type=str, required=True,
                           choices=['goal', 'missing_premises', 'valid_step'],
                           help='Name of the annotation labels you want to visualize.')
    argparser.add_argument('--title', '-t', type=str, help="Title of the graph", default='')
    argparser.add_argument('--x_axis_title', '-xat', type=str, help='Title of the X-Axis', default='')
    argparser.add_argument('--y_axis_title', '-yat', type=str, help='Title of the Y-Axis', default='')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--normalize', '-n', dest='normalize', action='store_true',
                           help='Normalize class counts between 0 and 1')

    args = argparser.parse_args()

    input_file: Path = Path(args.input_file)
    annotation_types: List[str] = [args.annotation_type]  # TODO: allow for multiple annotation types
    title: str = args.title
    x_axis_title: str = args.x_axis_title
    y_axis_title: str = args.y_axis_title
    output_file: Path = Path(args.output_file)
    force_output: bool = args.force_output
    normalize: bool = args.normalize

    # VALIDATE ARGS
    assert input_file.exists() and (str(input_file).endswith('.json') or str(input_file).endswith('.jsonl')), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # LOAD UP TREES
    json_trees = json.load(input_file.open('r'))
    trees = [Tree.from_json(t) for t in json_trees]

    annotation_scores = {k: {} for k in annotation_types}

    for tree in trees:
        for intermediate in tree.intermediates:
            annotations = intermediate.annotations
            for annotation_type in annotation_types:
                labels = annotations.get(annotation_type)

                if not labels:
                    continue

                if not isinstance(labels, list) and not isinstance(labels, tuple):
                    labels = [labels]

                for label in labels:
                    current_label_score = annotation_scores[annotation_type].get(label, 0)
                    annotation_scores[annotation_type][label] = current_label_score + 1

        for hypothesis in tree.hypotheses:
            annotations = hypothesis.annotations
            for annotation_type in annotation_types:
                labels = annotations.get(annotation_type)

                if not labels:
                    continue

                if not isinstance(labels, list) and not isinstance(labels, tuple):
                    labels = [labels]

                for label in labels:
                    current_label_score = annotation_scores[annotation_type].get(label, 0)
                    annotation_scores[annotation_type][label] = current_label_score + 1

    fig = plt.figure(figsize=(10, 5))

    legend = []

    x = []
    y = []

    for annotation_type in annotation_types:
        for idx, label in enumerate(annotation_scores[annotation_type]):
            x.append(f'{annotation_type}: {label}')
            y.append(annotation_scores[annotation_type][label])

    if normalize:
        y = [val / sum(y) for val in y]

    # Sort the x and y values
    (x, y) = zip(*(sorted(zip(x, y), key=lambda dx: dx)))

    for i in range(len(x)):
        plt.bar(x[i], y[i], color=cm.jet(1.*i/len(x)))



    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title(title)
    plt.legend()

    plt.savefig(str(output_file))
