import sys
import time
from pathlib import Path
import subprocess
import argparse
import shutil

ROOT_FOLDER = Path(__file__).parent
TRAINED_MODELS_FOLDER = ROOT_FOLDER / 'trained_models'
TMP_FOLDER = TRAINED_MODELS_FOLDER / 'tmp'

TRAINED_MODELS_FOLDER.mkdir(exist_ok=True, parents=True)
TMP_FOLDER.mkdir(exist_ok=True, parents=True)

files = {
    'abductive_heuristic': {'id':         'c0y83ygsqy6iz3gfxnw5nrocab6xy3es', 'name': 'abductive_h', 'local_filename': 'abductive_gc'},
    'deductive_heuristic': {'id':         'uyhxtbsux6kobpf7bmkpsvp2s8c0ksda', 'name': 'forward_h', 'local_filename': 'forward_v3_gc'},
    'abductive_step_model_small': {'id':  'hk29p9e2cjzghi1z5yrgj2w4d5jui66s', 'name': 't5_abductive_step'},
    'deductive_step_model_small': {'id':  'dkwsno6mrzudaysre8ggbmk3dicz90b8', 'name': 't5_large_pps_eb_step'},
    'wanli_entailment_model': {'id':      '9dupnp2rkcvtor3pikea3k1clc3b5qie', 'name': 'wanli_entailment_model'},
    't5_3b_abductive_eb_only': {'id':     '4v1sldwgmx01lkkghphsx7beghbxckpr', 'name': 't5_3b_abductive_eb_only'},
    't5_3b_eb_only_all_step': {'id':      'l1urbhdb1vhuiyp5hspx8sgvsrdxkw0b', 'name': 't5_3b_eb_only_all_step'}
}


def install_file(file_id, name, local_name: str = None):
    """
    Install a file from UT Box.  The file is downloaded to the {ROOT_FOLDER}/trained_models/tmp folder.  It is then 
    unzipped into the {ROOT_FOLDER}/trained_models folder.

    Then, we check and remove a folder called __MACOSX  (something that was put in on the original upload, but not 
    necessary for using the model).
    
    :param file_id: The BOX file id for the current file being downloaded.
    :param name: The name of the file in the BOX folder (as well as the name of the folder to be used locally)
    :return: N/A
    """

    subprocess.run(['curl', '-L', f'https://utexas.box.com/shared/static/{file_id}.zip', '--output', str(TMP_FOLDER / f'{name}.zip')])
    subprocess.run(['unzip', str(TMP_FOLDER / f'{name}.zip'), '-d', str(TRAINED_MODELS_FOLDER)])
    if local_name:
        subprocess.run(['mv', str(TRAINED_MODELS_FOLDER / name), str(TRAINED_MODELS_FOLDER / local_name)])
    subprocess.run(['rm', '-r', str(TRAINED_MODELS_FOLDER / '__MACOSX')])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ignore_list', '-i', choices=list(files.keys()), type=str, nargs='+',
        help='Any file you may not want to download and put into the trained_folders folder.'
    )
    parser.add_argument(
        '--allow_list', '-a', choices=list(files.keys()), type=str, nargs='+',
        help='Any of the files you want to download and put into the trained_folders folder.'
    )
    parser.add_argument('--show_files', '-s', action='store_true', help='List available files to download.')

    args = parser.parse_args()

    ignore_list = args.ignore_list
    allow_list = args.allow_list
    show_files = args.show_files

    if show_files:
        print("==== Files you can download are ===")
        for f in files.keys():
            print(f'\t{f}')
        sys.exit(0)

    files_to_download = list(files.keys())
    if ignore_list is not None:
        files_to_download = [x for x in files_to_download if x not in ignore_list]
    if allow_list is not None:
        files_to_download = [x for x in files_to_download if x in allow_list]

    assert len(files_to_download) > 0, \
        'No files were found using the ignore and allow list parameters!'

    print('=====   INFO   ======')
    print("These are big files and it may take awhile to download.")
    print('=====================')

    for file in files_to_download:
        install_file(files[file]['id'], files[file]['name'], files[file].get('local_filename', None))

    if TMP_FOLDER.exists():
        shutil.rmtree(str(TMP_FOLDER))

