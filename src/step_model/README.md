Overview
-

This folder contains the scripts needed to train a step model (both deductive and abductive) using s2s formatted
data (check the `src/data_scripts` folder for more on s2s formats)

Training
-

```shell
python train_step_model.py --output_dir ../../trained_models/test --data_train ../../data/step/entailmentbank/entailmentbank_train_clean_s2s/train --data_eval ../../data/step/entailmentbank/entailmentbank_dev_s2s/val
```

Here the `--output_dir` the path to the trained_models folder and the name of the model (this will be used in the `configs` folder when you run experiments in `src/experiments` so pick a good name)

`--data_train` is the path to the step s2s data for the model (if you are training an abductive model make sure you run on abductive s2s data, check `src/data_scripts` for more info)

`--data_eval` is the same as data_train except this is for validation data to run per epoch
