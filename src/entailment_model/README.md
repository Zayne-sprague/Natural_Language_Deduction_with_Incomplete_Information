Entailment Models
---

Note: this was a separate project originally isolated from the main work; which is why some of this is disjoint.

It was also originally called "evaluation" (evaluating the entailment of two sentences) but we changed it to "entailment_model"
so if you see "evaluation" in this folder -- its really just "entailment". 


### How to setup (from the root directory)

1.) Create a virtual env

`virtualenv venv`

2.) Install the pip reqs with 

`pip install -r requirements.txt`

3.) Navigate into the project folder

`cd src/entailment_model`

4.) Run the install script (downloads the Transformer runner from Kaj)

`./install.sh`

Notes:

This will try to set your python path, if it doesn't get set, copy the line manually and run it within the evaluation folder

`export PYTHONPATH=$PYTHONPATH:$PWD/fp_dataset_artifacts/:$PWD/..`

This will also uninstall the `transformers` pip package and install the one specified in the fp-dataset-artifacts repo

If you see an error about the python package `datasets`, run the install script again.

YOU WILL have to rerun the requirements.txt and stuff for the other parts of the code-base


## Scripts

`generate_dataset.py` - will create a dataset to use for training

`train.py` - will train a model given a dataset name 

`evaluate.py` - will generate metrics on how well the trained model does

`entropy_scores.py` - will generate a .csv with entropy values for every prediction


## Example run

```shell
python generate_dataset.py --validation_percentage=0.33 --folded

python train.py --epochs=2

python evaluate.py --test_all_thresholds --export_bad_predictions

python entropy_scores.py
```

There are a lot of parameters you can pass into these scripts, the most important ones being

`dataset_name` and `run_name` if you do not specify these, the model you train and the dataset you generate
will be overwritten the next time someone runs the command (the default for these params will create temporary folders)

If you specify a `dataset_name` or a `run_name` the other scripts will also take them as parameters and use those 
instead of the default `tmp_dataset/tmp_model` values, they will also never be overwritten if they are not equal to 
`tmp_dataset` or `tmp_model`
