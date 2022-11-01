from argparse import ArgumentParser, Namespace
import time
from pathlib import Path
import shutil
import subprocess
import sys
from subprocess import Popen, PIPE, CalledProcessError
import json
from typing import List, Dict
import random
import datasets
datasets.disable_progress_bar()

import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import yaml

from utils.paths import SEARCH_CONFIGS_FOLDER, SEARCH_OUTPUT_FOLDER, SEARCH_DATA_FOLDER
from utils.config_handler import merge_yaml_and_namespace
from search.tree import Tree
from scripts.create_shallow_trees import create_shallow_trees
from scripts.create_premise_masked_trees import create_premise_masked_trees
from scripts.search import search
from scripts.score_searches import score_searches
from scripts.find_tree_proofs import find_tree_proofs
from scripts.score_proofs import score_proofs
from scripts.convert_to_csv import convert_to_csv
from scripts.recover_premise_csv import recover_premise_csv


def track_progress(trackfile: Path, progress: Dict[str, any]):
    with trackfile.open('w') as file:
        json.dump(progress, file)


def get_progress(trackfile: Path):
    if not trackfile.exists():
        return {}

    with trackfile.open('r') as file:
        return json.load(file)


def reset_progress(progress: Dict[str, any], reset_to: str = None):
    ckpts = [
        'init',
        'data_created',
        'searched',
        'scored',
        'recover_premise_csv',
        'find_proof_trees',
        'score_proof_trees',
        'proofs_to_csv'
    ]

    start_idx = -1 if reset_to is None else ckpts.index(reset_to)

    for i in range(start_idx, len(ckpts)):
        progress[ckpts[i]] = False

    return progress


if __name__ == "__main__":

    argparser = ArgumentParser()
    timestamp = str(time.time()).replace('.', '')

    argparser.add_argument('--max_trees', '-mt', type=int, default=-1,
                           help='Maximum number of trees to search over')
    argparser.add_argument('--search_types', '-s', type=str, nargs='+',
                           choices=['abductive', 'forward', 'abductive_and_forward'],
                           default=['abductive_and_forward'],
                           help='Types of searches to experiment on.')
    argparser.add_argument('--experiment_root_directory', '-erd', type=str, default=str(SEARCH_OUTPUT_FOLDER),
                           help='The root directory to save the experiment and its outputs.')
    argparser.add_argument('--experiment_name', '-en', type=str, default=f'm_exp_{timestamp}',
                           help='Name of the experiment (folder everything will be saved under in the '
                                '--experiment_root_directory.')

    argparser.add_argument('--force_output', '-f', dest='force_output', action='store_true',
                           help='Overwrite anything currently written in the output file path.')
    argparser.add_argument('--resume', '-r', dest='resume', action='store_true',
                           help='Resume experiment')
    argparser.add_argument('--resume_at', '-ra', type=str,
                           help='If --resume is set, this will determine where to resume the exp at')
    argparser.add_argument('--seed', '-sd', type=int, default=123,
                           help='Use this to set the random seed')

    args = argparser.parse_args()

    max_trees: int = args.max_trees
    search_types: List[str] = args.search_types
    root_dir: Path = Path(args.experiment_root_directory)
    experiment_name: str = args.experiment_name
    force_output: bool = args.force_output
    resume: bool = args.resume
    resume_at: str = args.resume_at
    seed: int = args.seed

    experiment_folder = root_dir / experiment_name
    data_dir = experiment_folder / 'data'
    output_dir = experiment_folder / 'output'
    config_dir = experiment_folder / 'config'
    vis_dir = experiment_folder / 'visualizations'

    assert not experiment_folder.exists() or force_output, \
        'Please specify an empty file path for the output parameter -o OR specify the force flag -f'

    # Overwrite the existing experiment (delete everything)
    if experiment_folder.exists() and not resume:
        shutil.rmtree(str(experiment_folder))

    experiment_folder.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    vis_dir.mkdir(exist_ok=True, parents=True)
    config_dir.mkdir(exist_ok=True, parents=True)

    trackfile = data_dir / 'progress.json'
    progress = get_progress(trackfile)
    if resume:
        progress = reset_progress(progress, reset_to=resume_at)

    orig_config_file = SEARCH_CONFIGS_FOLDER / 'enwn.yaml'
    config_file = config_dir / 'config.yaml'

    orig_data_file = SEARCH_DATA_FOLDER / 'enwn_full.json'
    data_file = data_dir / 'raw_dataset__enwn_full.json'
    shallow_trees_data_file = data_dir / 'shallow_trees__m_task_1_test.json'
    missing_premise_data_file = data_dir / 'missing_premises__m_task_test.json'
    search_data_file = data_dir / 'tree_dataset.json'

    if not progress.get('started'):
        shutil.copyfile(str(orig_data_file), str(data_file))

        progress['init'] = True
        track_progress(trackfile, progress)

    shutil.copyfile(str(orig_config_file), str(config_file))

    if not progress.get('data_created'):
        data_args = Namespace()
        data_args = merge_yaml_and_namespace(config_file, data_args, ['default.create_shallow_trees'])

        create_shallow_trees(
            input_file=data_file,
            output_file=shallow_trees_data_file,
            force_output=True,
            **vars(data_args)
        )

        data_args = Namespace()
        data_args = merge_yaml_and_namespace(config_file, data_args, ['default.create_premise_masked_trees'])

        create_premise_masked_trees(
            input_file=shallow_trees_data_file,
            output_file=missing_premise_data_file,
            force_output=True,
            **vars(data_args)
        )

        # LOAD UP TREES
        random.seed(seed)

        json_trees = json.load(missing_premise_data_file.open('r'))
        trees = [Tree.from_json(t) for t in json_trees]
        if max_trees > -1:
            rand_indices = random.sample(range(0, len(trees)), min(max_trees, len(trees)))
            trees = [trees[x] for x in rand_indices]

        with search_data_file.open('w') as file:
            json.dump([x.to_json() for x in trees], file)

        progress['data_created'] = True
        track_progress(trackfile, progress)

    if not progress.get('searched'):

        if 'abductive' in search_types:
            search_args = Namespace()
            search_args = merge_yaml_and_namespace(config_file, search_args, ['abductive_only.search'], 'default.search')

            search(
                input_file=search_data_file,
                output_file=output_dir / 'raw_search__abductive_only.json',
                force_output=True,
                **vars(search_args)
            )

        if 'forward' in search_types:
            search_args = Namespace()
            search_args = merge_yaml_and_namespace(config_file, search_args, ['forward_only.search'], 'default.search')

            search(
                input_file=search_data_file,
                output_file=output_dir / 'raw_search__forward_only.json',
                force_output=True,
                **vars(search_args)
            )

        if 'abductive_and_forward' in search_types:
            search_args = Namespace()
            search_args = merge_yaml_and_namespace(config_file, search_args, ['abductive_and_forward.search'], 'default.search')

            search(
                input_file=search_data_file,
                output_file=output_dir / 'raw_search__abductive_and_forward.json',
                force_output=True,
                **vars(search_args)
            )

        progress['searched'] = True
        track_progress(trackfile, progress)

    if not progress.get('scored'):

        if 'abductive' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['abductive_only.score_searches'], 'default.score_searches')

            score_searches(
                input_file=output_dir / 'raw_search__abductive_only.json',
                output_file=output_dir / 'scored_search__abductive_only.json',
                force_output=True,
                **vars(score_args)
            )

        if 'forward' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['forward_only.score_searches'], 'default.score_searches')

            score_searches(
                input_file=output_dir / 'raw_search__forward_only.json',
                output_file=output_dir / 'scored_search__forward_only.json',
                force_output=True,
                **vars(score_args)
            )

        if 'abductive_and_forward' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['abductive_and_forward.score_searches'], 'default.score_searches')

            score_searches(
                input_file=output_dir / 'raw_search__abductive_and_forward.json',
                output_file=output_dir / 'scored_search__abductive_and_forward.json',
                force_output=True,
                **vars(score_args)
            )

        progress['scored'] = True
        track_progress(trackfile, progress)

    if not progress.get('recover_premise_csv'):

        if 'abductive' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['abductive_only.recover_premise_csv'], 'default.recover_premise_csv')

            recover_premise_csv(
                input_file=output_dir / 'scored_search__abductive_only.json',
                output_file=output_dir / 'recovered_premise__abductive_only.csv',
                force_output=True,
                **vars(score_args)
            )

        if 'forward' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['forward_only.recover_premise_csv'], 'default.recover_premise_csv')

            recover_premise_csv(
                input_file=output_dir / 'scored_search__forward_only.json',
                output_file=output_dir / 'recovered_premise__forward_only.csv',
                force_output=True,
                **vars(score_args)
            )

        if 'abductive_and_forward' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['abductive_and_forward.recover_premise_csv'], 'default.recover_premise_csv')

            recover_premise_csv(
                input_file=output_dir / 'scored_search__abductive_and_forward.json',
                output_file=output_dir / 'recovered_premise__abductive_and_forward.csv',
                force_output=True,
                **vars(score_args)
            )

        progress['recover_premise_csv'] = True
        track_progress(trackfile, progress)

    if not progress.get('find_proof_trees'):

        if 'abductive' in search_types:
            proof_args = Namespace()
            proof_args = merge_yaml_and_namespace(config_file, proof_args, ['abductive_only.find_tree_proofs'], 'default.find_tree_proofs')

            find_tree_proofs(
                input_file=output_dir / 'scored_search__abductive_only.json',
                output_file=output_dir / 'tree_proofs__abductive_only.json',
                force_output=True,
                **vars(proof_args)
            )

        if 'forward' in search_types:
            proof_args = Namespace()
            proof_args = merge_yaml_and_namespace(config_file, proof_args, ['forward_only.find_tree_proofs'], 'default.find_tree_proofs')

            find_tree_proofs(
                input_file=output_dir / 'scored_search__forward_only.json',
                output_file=output_dir / 'tree_proofs__forward_only.json',
                force_output=True,
                **vars(proof_args)
            )

        if 'abductive_and_forward' in search_types:
            proof_args = Namespace()
            proof_args = merge_yaml_and_namespace(config_file, proof_args, ['abductive_and_forward.find_tree_proofs'],
                                                  'default.find_tree_proofs')

            find_tree_proofs(
                input_file=output_dir / 'scored_search__abductive_and_forward.json',
                output_file=output_dir / 'tree_proofs__abductive_and_forward.json',
                force_output=True,
                **vars(proof_args)
            )

        progress['find_proof_trees'] = True
        track_progress(trackfile, progress)

    if not progress.get('score_proof_trees'):

        if 'abductive' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['abductive_only.score_proofs'], 'default.score_proofs')

            score_proofs(
                input_file=output_dir / 'tree_proofs__abductive_only.json',
                output_file=output_dir / 'scored_tree_proofs__abductive_only.json',
                force_output=True,
                **vars(score_args)
            )


        if 'forward' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['forward_only.score_proofs'], 'default.score_proofs')

            score_proofs(
                input_file=output_dir / 'tree_proofs__forward_only.json',
                output_file=output_dir / 'scored_tree_proofs__forward_only.json',
                force_output=True,
                **vars(score_args)
            )

        if 'abductive_and_forward' in search_types:
            score_args = Namespace()
            score_args = merge_yaml_and_namespace(config_file, score_args, ['abductive_and_forward.score_proofs'], 'default.score_proofs')

            score_proofs(
                input_file=output_dir / 'tree_proofs__abductive_and_forward.json',
                output_file=output_dir / 'scored_tree_proofs__abductive_and_forward.json',
                force_output=True,
                **vars(score_args)
            )

        progress['score_proof_trees'] = True
        track_progress(trackfile, progress)

    if not progress.get('proofs_to_csv'):

        if 'abductive' in search_types:
            export_args = Namespace()
            export_args = merge_yaml_and_namespace(config_file, export_args, ['abductive_only.convert_to_csv'], 'default.convert_to_csv')

            convert_to_csv(
                input_file=output_dir / 'scored_tree_proofs__abductive_only.json',
                output_file=output_dir / 'proofs__abductive_only.csv',
                force_output=True,
                **vars(export_args)
            )

        if 'forward' in search_types:
            export_args = Namespace()
            export_args = merge_yaml_and_namespace(config_file, export_args, ['forward_only.convert_to_csv'], 'default.convert_to_csv')

            convert_to_csv(
                input_file=output_dir / 'scored_tree_proofs__forward_only.json',
                output_file=output_dir / 'proofs__forward_only.csv',
                force_output=True,
                **vars(export_args)
            )

        if 'abductive_and_forward' in search_types:
            export_args = Namespace()
            export_args = merge_yaml_and_namespace(config_file, export_args, ['abductive_and_forward.convert_to_csv'], 'default.convert_to_csv')

            convert_to_csv(
                input_file=output_dir / 'scored_tree_proofs__abductive_and_forward.json',
                output_file=output_dir / 'proofs__abductive_and_forward.csv',
                force_output=True,
                **vars(export_args)
            )

        progress['proofs_to_csv'] = True
        track_progress(trackfile, progress)
