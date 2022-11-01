Overview
-

This folder brings together a lot of the work in `src/scripts` and `src/search` to allow for an easy way to run a
dataset through our search algorithm the same way we did in our paper.

There are multiple steps an experiment will go through, 

1.) data creation

2.) Search and generation

3.) Proof search

4.) Proof scoring

These all roughly line up with checkpoints in the `eb_experiment` and `enwn_experiment` files if you want to check the actual names.  But 
by breaking up the experiment this way, it's very easy to resume an experiment or rerun specific parts of an experiemnt without 
rerunning the whole thing (usually search, step 2, takes the longest).

The output is sent to `{PROJECT_ROOT}/output` and is supposed to be _fully contained_ meaning the data, output, proofs, etc.
are all copied into a single folder in the output directory for easy reproducability. 

Running an experiment
-

```shell
python eb_experiment.py -en initial_eb_experiment -mt 100 
```

Here the param `-en` is the experiment name, where the output will be stored (`{ROOT_DIRECTORY}/initial_eb_experiment` for this example)

`-mt` is the number of trees to run (maximum trees)

There are a ton of other parameters to help guide the experiment along, you can read about them via

```shell
python eb_experiment.py -h
```

Configs
-

Configs define how the data creation, search, scoring, and proof scoring should operate.  There are a ton of
parameters so we created yamls instead of inline them all as command line arguments.  Most should be self explanatory
however, any questions about what an argument means can be found in their respective script I.E. search related 
arguments are heavily documented in `src/scripts/search.py`

`eb_experiment.py` uses the config `{ROOT}/configs/entailment_bank.yaml`

`enwn.py` uses the config `{ROOT}/configs/enwn.yaml`



Making my own dataset
-


Most of the differences between the two scripts is really just the dataset file location and the config name so
if you need to make your own experiment with a custom dataset, copying and pasting one of the existing experiments
should be the first step