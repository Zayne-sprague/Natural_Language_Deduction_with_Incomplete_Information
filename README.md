# Natural Language Deduction with Incomplete Information


<image src="./images/overview.png"></image>

Generating premises in underspecified settings through abduction.

https://arxiv.org/abs/2211.00614

## Overview

This repository holds the code for the paper _Natural Language Deduction with Incomplete Information_

All the data for training and evaluating models can be found in `{PROJECT_ROOT}/data`

The newly introduced **Everyday Norms: Why Not?** dataset can be found in `{PROJECT_ROOT}/data/full/enwn/enwn.json`
(to visualize it check the README in `src/visualization_tools`)

For examples on how to call our Step Models, Heuristics, etc. check `{PROJECT_ROOT}/examples.ipynb` for a brief overview.

Otherwise, all the major sections of our system found in `{PROJECT_ROOT}/src` are either heavily documented or have a
README inside their root folder meant to help guide the reader towards using our code.

## Checkpoints & Data

Checkpoints are coming soon as well as the version of the WANLI dataset used for the entailment model.

The entailmentbank and Everday Norms: Why Not? datasets can be found in `data/full`

## Installation

1. `virtualenv venv` we have tested with python 3.8
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python ./install.py` (expect this to take a long time based on your internet speed)

Step 4 will download all the models needed to recreate the experiments from the paper as well as two small step models
that also perform well.  (We use the smaller step models for testing/debugging and the 3billion parameter models for
experiments.)

You can see all the models that can be downloaded via
`python ./install.py -s` 

And you can install them individually via
`python ./install.py -a abductive_heuristic abductive_step_model_small ...etc`

## Getting started (recreate paper results)

1. Follow the `examples.ipynb`
2. Go to `src/experiments` and follow the README
3. to recreate the paper results make sure to update the config folder with the right parameters.


## Folder breakdown
- **configs**: parameters used for experiments (templates for the configs of the experiments we used in our paper)
- **data**: All training and evaluation data we used in the paper
- **output**: where the experiments are set to put the data/output/results of a run (check `src/experiments`)
- **src**: All source code for our project broken up into submodules.
- **trained_models**: Important folder where all trained models should be stored **(put any checkpoints downloaded into this folder)**

## Requirements
All experiments were performed on the following specs (some may not matter)

- Python Version: 3.8.3
- tested on Mac and Linux
- Transformers version 4.20.0 (really important)

Make sure you run `pip install -r requirements.txt`

### Authors
- Zayne Sprague
- Kaj Bostrom
- Swarat Chaduri
- Greg Durrett


