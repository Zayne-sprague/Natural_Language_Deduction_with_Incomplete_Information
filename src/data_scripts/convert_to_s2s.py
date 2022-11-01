import json
import sys
import random
from pathlib import Path
import argparse
from jsonlines import jsonlines

from data_scripts.data_utils import write_dataset_files
from utils.paths import DATA_FULL_FOLDER


def normalize(sent):
    ns = sent.replace(" ,", ",").replace(" '", "'").strip()
    if ns[-1] != ".":
        ns += "."
    return ns[0].upper() + ns[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dataset_name",
        "-odn",
        type=str,
        help="The path and name of the output dataset i.e. " '"enwn/val"',
    )

    parser.add_argument(
        "--input_file",
        "-if",
        type=str,
        help="Input file within the data/full/ directory",
    )

    parser.add_argument(
        "--use_intermediates",
        "-ui",
        action="store_true",
        dest="use_intermediates",
        help="Use the intermediates as the targets and the inputs to generate them as their premises",
    )

    parser.add_argument(
        "--overwrite_output",
        "-oo",
        action="store_true",
        dest="overwrite_output",
        help="If the output dataset exists",
    )

    args = parser.parse_args()

    output_dataset_name = args.output_dataset_name
    input_file = args.input_file
    use_intermediates = args.use_intermediates
    overwrite_output = args.overwrite_output

    if input_file.endswith(".jsonl"):
        source_data = list(jsonlines.open(str(DATA_FULL_FOLDER / input_file), "r"))
    else:
        source_data = json.load((DATA_FULL_FOLDER / input_file).open("r"))

    premise_lines = []
    target_lines = []

    # TODO - make the use_intermediate flag try to use the premises that made the intermediate step output not the
    #  actual target

    for ex in source_data:
        premises = ex["premises"]
        intermediates = ex["intermediates"]

        intermediate_outputs = [normalize(x["output"]) for x in intermediates]

        if use_intermediates:
            for step in intermediates:
                ins = step["inputs"]

                ins_for_step = []

                for _in in ins:
                    idx = int(_in[1:])

                    if _in.startswith("p"):
                        ins_for_step.append(premises[idx])
                    else:
                        ins_for_step.append(intermediate_outputs[idx])

                target = normalize(step["output"])

                premise_lines.append(" ".join(ins_for_step))
                target_lines.append(target)

        else:
            target = normalize(ex["hypothesis"])

            premises = ex["premises"]
            premises = [normalize(x) for x in premises]

            premise_lines.append(" ".join(premises))
            target_lines.append(target)

    write_dataset_files(
        output_dataset_name,
        premise_lines,
        target_lines,
        overwrite_existing=overwrite_output,
    )
