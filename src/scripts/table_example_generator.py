from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple
import json
from jsonlines import jsonlines
from tqdm import tqdm
import csv
import random
random.seed(1)

from utils.paths import SEARCH_DATA_FOLDER
from search.tree.tree import Tree, normalize
from search.tree.step_generation import StepGeneration
from utils.paths import SEARCH_OUTPUT_FOLDER


def proof_reports(
        input_files: List[Path],
        input_experiments: List[Path],
):
    """
    Wrapper that can export a list of trees from a file or a list of trees where each tree is a list of proofs for that
    tree.

    :param input_files: Paths to files with trees to search over
    :param input_experiments: Paths to experiment folders with searches
    """

    tree_sets = []
    file_names = []
    data_file_names = []
    data_sets = []
    search_file_names = []
    search_sets = []

    for input_file in input_files:
        # LOAD UP TREES
        if str(input_file).endswith('.jsonl'):
            tree_sets.append(list(jsonlines.open(str(input_file), 'r')))
            file_names.append(input_file)
        else:
            tree_sets.append(json.load(input_file.open('r')))
            file_names.append(input_file)

    for experiment in input_experiments:
        scored_searches = experiment / 'output'

        for file in scored_searches.glob('tree_proofs*.json'):
            tree_sets.append(json.load(file.open('r')))
            file_names.append(str(file))

        for file in (experiment / 'output').glob('scored_search*.json'):
            data_sets.append([Tree.from_json(x) for x in json.load(file.open('r'))])
            data_file_names.append(str(file))

        for file in (experiment / 'output').glob('scored_search*'):
            search_file_names.append(str(file))
            search_sets.append([Tree.from_json(x) for x in json.load(file.open('r'))])

    if len(tree_sets) == 0:
        return

    goals = {}
    proofs = {}
    for idx, tree_set in enumerate(tree_sets):
        proofs[file_names[idx]] = [[Tree.from_json(y) for y in x] for x in tree_set]

    for file_name in file_names:
        for idx, p in enumerate(proofs[file_name]):
            goals[p[0].missing_premises[0]] = goals.get(p[0].missing_premises[0], {})
            goals[p[0].missing_premises[0]][file_name] = p

    good_goals = {k: v for k, v in goals.items() if len(v.keys()) == len(file_names)}

    good_data = {}

    for mp in good_goals.keys():
        for idx, (p_filename, data) in enumerate(zip(good_goals[mp], data_sets)):
            for didx, d in enumerate(data):
                if d.missing_premises[0] == mp:
                    proofs = good_goals[mp][p_filename]
                    for pidx, p in enumerate(proofs):
                        # good_goals[mp][p_filename][pidx].input = [*list(set(p.normalized_premises).intersection(set(d.normalized_premises))), p.goal]
                        # good_goals[mp][p_filename][pidx].output = list(set(p.normalized_premises) - set(d.normalized_premises))
                        good_data[mp] = good_data.get(mp, {})
                        good_data[mp][p_filename] = good_data[mp].get(p_filename, [])

                        pdata = {
                            'input': [*list(set(p.normalized_premises).intersection(set(d.normalized_premises))), p.goal],
                            'output': list(set(p.normalized_premises) - set(d.normalized_premises))[0],
                            # 'score': p.intermediates[0].scores['forward_agreement']
                        }

                        scored_search = search_sets[idx][didx]

                        for h in scored_search.hypotheses:
                            if normalize(h.output) == pdata['output']:
                                pdata['score'] = h.scores['missing_premises'][0]
                                break

                        if 'score' not in pdata:
                            continue


                        good_data[mp][p_filename].append(pdata)
                    break

    # print(good_goals)
    good_data = {k: v for k, v in good_data.items() if all([len(x) > 0 for _, x in v.items()])}

    lst = list(good_data.items())
    _max = 100
    i = 0


    for mp, exps in random.sample(lst, len(lst)):
        # print(f'Missing Premise: {mp}')
        row_str = ''
        row_str = f'{mp} & '

        seen_input = False

        _types = []
        dupe_found = False

        for exp, proofs in exps.items():
            exp_name = Path(exp).parent.parent.name

            printed_name = False
            for p in proofs[0:min(1, len(proofs))]:
                if not seen_input:
                    p_data = " ".join(p['input'][0:-1])
                    # print(f"Premises: {p_data}")
                    # print(f"Goal: {p['input'][-1]}")
                    seen_input = True
                    # print()

                if not printed_name:
                    # print(exp_name)
                    printed_name = True

                # print(p['output'])
                row_str += f'{p["output"]} & {p["score"]:.2f} & '

                if p["output"] in _types:
                    dupe_found = True
                _types.append(p["output"])


            # print()

        # print(f'{mp}')

        # if dupe_found:
        #     continue
        print('\\midrule')
        print(row_str[0:-3] + ' \\\\')
        i += 1
        # print()

        if i >= _max:
            return


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('--input_files', '-i', type=str, nargs='+', default=[],
                           help='Path to file with trees to search over')
    argparser.add_argument('--input_experiments', '-e', type=str, nargs='+',
                           help='Path to experiment folders that contain the searched trees.')

    args = argparser.parse_args()

    _input_files: List[Path] = [Path(x) for x in args.input_files]
    _input_experiments: List[Path] = [SEARCH_OUTPUT_FOLDER / x for x in args.input_experiments]

    proof_reports(
        input_files=_input_files,
        input_experiments=_input_experiments,
    )
