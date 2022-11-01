Overview
-

This folder is responsible for creating and training the Heuristic models used in our paper (section 3.2)

We use two heuristics, a deductive heuristic SCORE(p1 + p2 | g) and an abductive heuristic SCORE(p - g).

These score functions are implemented as Deep Learning Models.

Training
-

###Deductive Heuristic

To train the deductive heuristic call

```bash
python train_deductive {full_train_data} {full_eval_data} --model_name {name}
```

The "full" data means a training file in the format of the files in `{ROOT_PROJECT}/data/full/*`

For the paper we used

```bash
python train_deductive ../../data/full/entailmentbank/task_1/train.jsonl ../../data/full/entailmentbank/task_1/dev.jsonl --model_name deductive_heuristic --goal_conditioned
```

###Abductive Heuristic

To train the abductive heuristic call (very similar to the deductive model)

```bash
python train_deductive {full_train_data} {full_eval_data} --model_name {name}
```

The "full" data means a training file in the format of the files in `{ROOT_PROJECT}/data/full/*`

For the paper we used

```bash
python train_deductive ../../data/full/entailmentbank/task_1/train.jsonl ../../data/full/entailmentbank/task_1/dev.jsonl --model_name abductive_heuristic
```

###Other params for training

Each training script has helpers on the arguments you can pass through (for example, `--goal_conditioned`). 

###Data format

Currently, this only works with jsonl formatted files (shouldn't be hard to extend though)

#Inference

Check `src/search/steptype_modeled.py` for an example on how to call the trained model for inference.  This class
is what is used in the main search procedure as well.
