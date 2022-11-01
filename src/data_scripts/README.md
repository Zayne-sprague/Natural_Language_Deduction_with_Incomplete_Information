Overview
-

Step Models are trained using a s2s format (sentence-to-sentence).

S2S basically means there's an input file where each line is a set of sentences (for deduction it should be p1, p2, p3, p4..., pn), then there is an output file that contains the goal generation (for deduction this would be the goal) per line.

Creating a Deductive dataset file
-

```shell
python convert_to_s2s.py -odn testing/test -if entailmentbank/task_1/train.jsonl -oo
```

Here `-odn` is the output dataset name that will be written out to in the `{Project Root}/data/step/{odn}`

`-if` is the input file that exists inside `{Project Root/data/full/{if}`

`-oo` just means overwrite original if an output file with the same name exists.

There should be 2 files created in `{Project Root}/data/step/{odn}` called `{Project Root}/data/step/{odn}.source` (the input file) and `{Project Root}/data/step/{odn}.target` (the goal generation file)

NOTE: this command will create a deductive data file! To create an abductive file you have to create a deductive one then call `convert_s2s_forward_to_abductive.py`

Creating an abductive dataset file
-

```shell
-idn {testing/test} -odn {testing/abd_test}
```

Following from the deductive example, this command takes the output of the deductive script and creates a new
dataset file that has abductive input (p1, p2, ..., pn-1, g) and an abductive target (pn).
