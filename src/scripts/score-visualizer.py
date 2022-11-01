from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict
import json
from jsonlines import jsonlines
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.paths import SEARCH_DATA_FOLDER
from search.tree.tree import Tree


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_files', '-i', type=str, help='Path to file with trees to search over', nargs='+')
    argparser.add_argument('--output_file', '-o', type=str,
                           help='Path to the output file that the evaluations will be stored in', required=True)
    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--line_names', '-ln', nargs='+', type=str, required=True,
                           help='names of the lines corresponding to the input files (should equal in length)')
    argparser.add_argument('--title', '-t', type=str, help="Title of the graph", default='')
    argparser.add_argument('--x_axis_title', '-xat', type=str, help='Title of the X-Axis', default='')
    argparser.add_argument('--y_axis_title', '-yat', type=str, help='Title of the Y-Axis', default='')
    argparser.add_argument('--column_label', '-cl', type=str, nargs='+', default=[],
                           help='Name of the column to use as the X-Axis and reference in the input files.'
                                ' If each file has a unique column, make sure the list of labels matches the number of '
                                ' input files.')

    args = argparser.parse_args()

    input_files: List[Path] = [Path(x) for x in args.input_files]
    output_file: Path = Path(args.output_file)
    force_output: bool = args.force_output
    line_names: List[str] = args.line_names
    title: str = args.title
    x_axis_title: str = args.x_axis_title
    y_axis_title: str = args.y_axis_title
    column_label: List[str] = args.column_label

    # VALIDATE ARGS
    assert all([f.is_file() and str(f).endswith('.json') for f in input_files]), \
        'Please specify a correct path to a json file with an array of trees to run the search over.'
    assert not output_file.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'
    assert len(column_label) == len(input_files) or len(column_label) == 0, \
        'Specify the same number of column labels as input files (in order) or specify a single column label to be ' \
        'used for all files'

    lines = []
    x_ticks = []
    for input_file in input_files:
        line: Dict[str, float] = json.load(input_file.open('r'))
        [x_ticks.append(x) for x in line.keys() if x not in x_ticks]
        lines.append(line)

    x_ticks = list(sorted(x_ticks))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xloc = plt.MaxNLocator(10)
    ax.xaxis.set_major_locator(xloc)

    for line_idx, (line, line_name) in enumerate(zip(lines, line_names)):
        bins = [line.get(k, {}) for k in x_ticks]
        if len(column_label) == 1:
            bins = [x.get(column_label[0], 0) for x in bins]
        if len(column_label) > 1:
            bins = [x.get(column_label[line_idx], 0) for x in bins]

        plt.plot(x_ticks, bins, label=line_name)


    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.title(title)
    plt.legend()

    plt.savefig(str(output_file))
