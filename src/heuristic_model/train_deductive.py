from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import torch
from torch.utils.data import IterableDataset
from transformers.trainer_utils import EvalPrediction
import argparse
import json
import numpy as np

from heuristic_model.generate_data import generate_data
from utils.paths import TRAINED_MODELS_FOLDER


class CalibrationDataset(IterableDataset):
    def __init__(
        self,
        source_data_path,
        tokenizer,
        goal_conditioned=False,
        indirect_goals=False,
        seed=0,
    ):
        with open(source_data_path, encoding="utf-8") as source_data_file:
            self.source_data = [json.loads(l) for l in source_data_file]
        self.tokenizer = tokenizer
        self.goal_conditioned = goal_conditioned
        self.indirect_goals = indirect_goals
        self.seed = seed
        self.length = sum(len(ex["intermediates"]) for ex in self.source_data) * 2

    def __iter__(self):
        for ex in generate_data(
            self.source_data,
            goal_conditioned=self.goal_conditioned,
            indirect_goals=self.indirect_goals,
            seed=self.seed,
        ):
            inputs_encoded = self.tokenizer(
                ex["input_text"],
                text_pair=ex["goal"] if self.goal_conditioned else None,
                truncation=True,
                padding=True,
            )
            inputs_encoded["label"] = ex["label"]
            yield inputs_encoded

    def __len__(self):
        return self.length


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    hits = np.logical_and(labels, preds).sum()
    p = (hits / preds.sum()).item()
    r = (hits / labels.sum()).item()
    f1 = 2.0 * (p * r) / (p + r)
    acc = (labels == preds).mean().item()
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("train_data", type=str)
    argp.add_argument("eval_data", type=str)

    argp.add_argument(
        "--model_name", type=str, required=True,
        help="Name to save the model as under {PROJECT_ROOT}/trained_models/{model_name}"
    )
    argp.add_argument(
        "--goal_conditioned", action="store_true",
        help="Give the goal as input to the model (only makes sense for deduction)"
    )
    argp.add_argument(
        "--indirect_goals", action="store_true",
        help="If false only use the main goal of the tree for goal_conditioned; if True then any conclusion beyond the"
             "current step can be used as the \"conditioned goal\" (usually false)"
    )
    argp.add_argument(
        "--model", type=str, default="bert-base-uncased",
        help='Base hugging face model to use for training.'
    )
    argp.add_argument("--num_epochs", type=int, default=5)
    argp.add_argument("--lr", type=float, default=5e-5)
    args = argp.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = CalibrationDataset(
        args.train_data, tokenizer, args.goal_conditioned, args.indirect_goals
    )
    eval_dataset = list(
        CalibrationDataset(
            args.eval_data, tokenizer, args.goal_conditioned, args.indirect_goals
        )
    )
    training_args = TrainingArguments(
        output_dir=TRAINED_MODELS_FOLDER / args.model_name,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_total_limit=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()
