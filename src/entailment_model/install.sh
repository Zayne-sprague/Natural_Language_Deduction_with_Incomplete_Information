git clone https://github.com/gregdurrett/fp-dataset-artifacts.git fp_dataset_artifacts

pip uninstall transformers
pip install -r ./fp_dataset_artifacts/requirements.txt
pip install torch==1.10

export PYTHONPATH=$PYTHONPATH:$PWD/fp_dataset_artifacts/:$PWD/..:$PWD/../..