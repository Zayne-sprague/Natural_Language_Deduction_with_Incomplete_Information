import json
import sys
import random
from pathlib import Path
from typing import List

from utils.paths import DATA_FOLDER


def normalize(sent):
    ns = sent.replace(" ,", ",").replace(" '", "'").strip() + "."
    return ns[0].upper() + ns[1:]


def write_dataset_file(
    path: Path,
    lines: List[str],
    line_sep: str = "\n",
    overwrite_existing: bool = False,
) -> bool:
    """
    Helper function that returns True if it was able to write the lines to a dataset file or not.
    """

    if path.exists() and not overwrite_existing:
        print(
            f"ERROR: Tried overwriting {path} without using the overwrite_existing flag, skipping writing."
        )
        return False

    # When writing, make sure the folder path to the file exists (don't make the file itself a folder, hence use the
    # parent not the actual path.
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for line in lines:
            f.write(f"{line}{line_sep}")
    return True


if __name__ == "__main__":
    # TODO - hardcoded stuff should be generalized.
    with open(sys.argv[1], encoding="utf8") as source_file:
        out_source = (
            DATA_FOLDER / "step/abductive_single_shot/entailmentbank_train_clean_s2s/"
        )
        out_target = (
            DATA_FOLDER / "step/abductive_single_shot/entailmentbank_train_clean_s2s/"
        )

        out_target.mkdir(parents=True, exist_ok=True)
        out_source.mkdir(parents=True, exist_ok=True)

        out_source /= "train.source"
        out_target /= "train.target"

        sources = []
        targets = []

        for l in source_file:
            source_ex = json.loads(l)

            for p in source_ex["premises"]:
                inputs = " ".join(
                    [
                        *[x for x in source_ex["premises"] if x != p],
                        source_ex["hypothesis"],
                    ]
                )
                target = p

                sources.append(inputs)
                targets.append(target)

        write_dataset_file(out_source, sources, overwrite_existing=True)
        write_dataset_file(out_target, targets, overwrite_existing=True)
