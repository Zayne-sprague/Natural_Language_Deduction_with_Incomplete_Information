import argparse
import jsonlines
from typing import List, Tuple, Dict

from data_scripts.data_utils import load_dataset_files, split_lines, write_dataset_files


def create_abduction(
        split_source_lines: List[List[str]],
        target_lines: List[str]
) -> Tuple[List[List[str]], List[str]]:
    """
    Given a list of split source lines (source lines from a dataset file that has been split into sentences or premises)
    and a list of target lines, this function will create a new dataset where the source lines are all but 1 of the
    original split source lines plus the target line with the target being the held out split source lines sentence.

    I.E.

    split source = [[I love you, Love means to like, ...], ...] targets = [[I like you], ...]

    will produce a new dataset

    abductive source = [[I love you, ..., I like you], ...] abductive targets = [[Love means to like], ...]

    :param split_source_lines: The lines from the original datasets source file split into sentences
    :param target_lines: The lines from the original datasets target file
    :returns: A new set of lines for the source file and a new set of lines for the target file
    """

    abductive_split_source_lines = []
    abductive_target_lines = []

    for split_source_line, target_line in zip(split_source_lines, target_lines):

        for target_source_line in split_source_line:
            sources = set(split_source_line)
            sources.remove(target_source_line)
            sources = list(sources)
            # The original target is always appended to the end, TODO - is this the right way to do it?
            sources.append(target_line.lstrip().rstrip())

            abductive_split_source_lines.append(sources)
            abductive_target_lines.append(target_source_line)

    return abductive_split_source_lines, abductive_target_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dataset_name', '-odn', type=str, help='The path and name of the output dataset i.e. '
                                                                        '"abductive/entailmentbank_dev_s2s/val"', required=True)

    parser.add_argument('--input_dataset_name', '-idn', type=str,
                        help='The path and name of hte input dataset that will be'
                             ' converted into the new abductive format. I.E. '
                             'entailmentbank/entailmentbank_dev_s2s/val', required=True)

    parser.add_argument('--overwrite_output', '-oo', action='store_true', dest='overwrite_output', help='If the output '
                                                                                                        'dataset exists'
                                                                                                        'overwrite it.')

    args = parser.parse_args()

    output_dataset_name = args.output_dataset_name
    input_dataset_name = args.input_dataset_name
    overwrite_output = args.overwrite_output

    source_lines, target_lines = load_dataset_files(input_dataset_name)
    split_source_lines = split_lines(source_lines)

    abductive_split_source, abductive_targets = create_abduction(split_source_lines, target_lines)
    abductive_source = [" ".join(x) for x in abductive_split_source]

    write_dataset_files(
        output_dataset_name,
        abductive_source,
        abductive_targets,
        overwrite_existing=overwrite_output
    )
