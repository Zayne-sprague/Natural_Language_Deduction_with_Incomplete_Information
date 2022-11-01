from pathlib import Path
from typing import Tuple, List, Dict

from utils.paths import DATA_STEP_FOLDER


def load_dataset_file(path: Path) -> List[str]:
    """
    Load up a dataset file or return an empty array if nothing is there
    """

    if not path.exists() or not path.is_file():
        return []

    with open(str(path), mode='r') as f:
        lines = []
        for line in f:
            lines.append(line)
    return lines


def load_dataset_files(dataset_name: str) -> Tuple[List[str], List[str]]:
    """
    Given a name/path to a dataset i.e. "entailmentbank/entailmentbank_train_clean_s2s/train"
    return the source and target lines from each of the files.

    :param dataset_name: Name of the dataset excluding the .source and .target
    :returns: Tuple of arrays of the lines within the dataset files.
    """

    source_file = DATA_STEP_FOLDER / (dataset_name + '.source')
    target_file = DATA_STEP_FOLDER / (dataset_name + '.target')

    source_lines = load_dataset_file(source_file)
    target_lines = load_dataset_file(target_file)

    return source_lines, target_lines


def write_dataset_file(
        path: Path,
        lines: List[str],
        line_sep: str = '\n',
        overwrite_existing: bool = False,
) -> bool:
    """
    Helper function that returns True if it was able to write the lines to a dataset file or not.
    """

    if path.exists() and not overwrite_existing:
        print(f"ERROR: Tried overwriting {path} without using the overwrite_existing flag, skipping writing.")
        return False

    # When writing, make sure the folder path to the file exists (don't make the file itself a folder, hence use the
    # parent not the actual path.
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w') as f:
        for line in lines:
            f.write(f'{line}{line_sep}')
    return True


def write_dataset_files(
        dataset_name: str,
        source_lines: List[str],
        target_lines: List[str],
        overwrite_existing: bool = False
):
    """Helper function for writing lines to a datasets files."""
    source_file: Path = DATA_STEP_FOLDER / (dataset_name + '.source')
    target_file: Path = DATA_STEP_FOLDER / (dataset_name + '.target')

    write_dataset_file(source_file, source_lines, overwrite_existing=overwrite_existing)
    write_dataset_file(target_file, target_lines, overwrite_existing=overwrite_existing)


def split_lines(lines: List[str], sep: str = '.') -> List[List[str]]:
    """
    Small helper function to split a line by the sentences in it.
    TODO - may be effected by lines with ... in them or Dr. / Mr. etc. probably a better way to accomplish this
    """

    split_lines = []
    for line in lines:
        # Remove trailing white spaces and split
        splits = line.lstrip().rstrip().split(sep)

        # Add all the splits to the lines array if its not empty space (if x is a truthy check, '' is equal to false)
        split_lines.append([f'{x.lstrip().rstrip()}.' for x in splits if x])

    return split_lines
