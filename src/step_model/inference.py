import torch
torch.manual_seed(0)
import transformers
from pathlib import Path
import argparse
from csv import writer
from typing import Union, List
from datasets import Dataset

from utils.paths import TRAINED_MODELS_FOLDER


def get_model(model_name: str, max_length: int = 128, device=torch.device('cpu')):
    """
    Helper script for loading up a model with a checkpoint in the trained_models folder
    """

    model_path: Path = TRAINED_MODELS_FOLDER / model_name

    if not model_path.is_dir():
        print(f"ERROR: Could not find model at {model_path}")
        return

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    model = model.to(device)
    model.config.update({'max_length': max_length})
    return model


def get_tokenizer(model_name: str):
    """
    Helper script for loading up a tokenizer from a model with a checkpoint in the trained_models folder
    """

    model_path: Path = TRAINED_MODELS_FOLDER / model_name

    if not model_path.is_dir():
        print(f"ERROR: Could not find model at {model_path}")
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
    return tokenizer


def parse_prompt(prompt: Union[str, List[str]], tokenizer, return_tensors: str = 'pt'):
    """Helper for parsing string prompts into tokens"""
    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors=return_tensors, truncation=True)
    else:
        return tokenizer(prompt, return_tensors=return_tensors, truncation=True, padding=True)


def inference(tokens, model, device=torch.device('cpu'), num_return_sequences: int = 1):
    """Helper for generating tokens from a tokenized prompt and a model"""
    return model.generate(
        tokens['input_ids'].to(device),
        num_return_sequences=num_return_sequences,
        do_sample=num_return_sequences > 1
    )


def parse_output(output, tokenizer):
    """Helper for decoding the output of a model given a tokenizer."""
    return tokenizer.batch_decode(output)


def generate_outputs(prompts, tokenizer, model, device=torch.device('cpu')):
    for prompt in prompts:
        yield generate_output(prompt, tokenizer, model, device)

def generate_output(
        prompts: Union[List[str], str],
        tokenizer,
        model,
        device=torch.device('cpu'),
        num_return_sequences: int = 2,
        batch_size: int = 1
):
    if isinstance(prompts, str):

        tokens = parse_prompt(prompts, tokenizer)
        output = inference(tokens, model, device=device, num_return_sequences=num_return_sequences)
        decoded_output = parse_output(output, tokenizer)

    else:

        dataset = Dataset.from_dict({'prompts': prompts})
        dataset = dataset.map(lambda e: e, batched=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        output = []
        decoded_output = []

        for batch in dataloader:
            batch_tokens = parse_prompt(batch['prompts'], tokenizer)
            batch_output = inference(batch_tokens, model, device=device, num_return_sequences=num_return_sequences)
            batch_decoded_output = parse_output(batch_output, tokenizer)

            output.extend(batch_output)
            decoded_output.extend(batch_decoded_output)

    return output, decoded_output, prompts

def write_out_output(path: Path, outputs, prompts, targets=None, overwrite=False):
    if path.exists() and not overwrite:
        print("ERROR: output file already exists")
        return

    with path.open('w') as f:

        cols = ['prompt', 'output']
        if targets:
            cols.append('target')

        csv_writer = writer(f)


        csv_writer.writerow(cols)
        for idx in range(len(outputs)):
            prompt = prompts[idx]
            out = outputs[idx]

            row = [prompt, out]
            if targets:
                row.append(targets[idx])

            csv_writer.writerow(row)


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', '-mn', type=str, help='name of the model in the trained_models folder.')

    parser.add_argument('--prompts', '-p', type=str, help='prompt to pass to the model, you can add multiple'
                                                         'prompts by passing in multiple strings i.e. '
                                                         '"string one abc." "string two 123."', nargs="+")

    parser.add_argument('--prompts_file', '-f', type=str, help='File containing prompts to use')

    parser.add_argument('--device', '-d', type=str, help='torch device to use default cpu', default='cpu')

    parser.add_argument('--target_file', '-tf', type=str, help='Path to the targets file that shouldve been predicted '
                                                               'by the model.  Used only for when you want to export'
                                                               'the results via --output_file')

    parser.add_argument('--output_file', '-of', type=str, help='Name of the file to save the generations too')

    parser.add_argument('--overwrite_output', '-oo', action='store_true', dest='overwrite_output', help='if the file'
                                                                                                        'for the output'
                                                                                                        'exists, '
                                                                                                        'overwrite it.')

    parser.add_argument('--max_preds', '-mp', type=int, help='Only do the first N predictions', default=-1)

    parser.add_argument('--silent', '-s', action='store_true', dest='silent', help='Do not print out the predictions.')

    parser.add_argument('--max_output_length', '-mol', type=int, help='Max length of the generated output', default=128)


    args = parser.parse_args()

    _model_name = args.model_name
    _prompts = args.prompts
    _prompts_file = args.prompts_file
    _device = args.device
    _output_file = args.output_file
    _target_file = args.target_file
    _max_preds = args.max_preds
    _silent = args.silent
    _overwrite_output = args.overwrite_output
    _max_output_length = args.max_output_length

    # Load up every line as a new prompt from the text file.
    if not _prompts and _prompts_file:
        _prompts = []
        path = Path(_prompts_file)
        if path.exists() and path.is_file():
            with path.open('r') as f:
                for line in f:
                    _prompts.append(line)

    if len(_prompts) == 0:
        print("ERROR: No prompts were found/specified.")
        sys.exit(-1)

    _torch_device = torch.device(_device)
    _model = get_model(_model_name, max_length=_max_output_length, device=_torch_device)
    _tokenizer = get_tokenizer(_model_name)


    _outs = []
    _ps = []
    _ts = []
    for idx, (_, output, p) in enumerate(generate_outputs(_prompts, _tokenizer, _model, _torch_device)):
        output = output[0]
        p = p.lstrip().rstrip()

        _outs.append(output)
        _ps.append(p)

        if not _silent:
            print(f'Prompt: {p}\nGenerated: {output}\n')

        if _max_preds > -1 and idx > _max_preds:
            break

    if _target_file:
        with Path(_target_file).open('r') as f:
            for line in f:
                _ts.append(line.rstrip().lstrip())

    if _output_file:
        write_out_output(Path(_output_file), _outs, _ps, targets=_ts, overwrite=_overwrite_output)

