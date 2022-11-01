from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser, DataCollatorForSeq2Seq
import torch
import datasets

def load_raw_dataset(path_prefix):
    input_texts = []
    target_texts = []
    with open(path_prefix+'.source', encoding='utf-8') as data_file:
        for line in data_file:
            input_texts.append(line.strip())
    with open(path_prefix+'.target', encoding='utf-8') as data_file:
        for line in data_file:
            target_texts.append(line.strip())

    return datasets.Dataset.from_dict({'input_text': input_texts, 'target_text': target_texts})


if __name__ == "__main__":
    argp = HfArgumentParser(Seq2SeqTrainingArguments)
    argp.add_argument('--model', type=str, default='t5-large')
    argp.add_argument('--data_train', type=str, required=True)
    argp.add_argument('--data_eval', type=str)
    argp.add_argument('--max_input_length', type=int, default=128)
    argp.add_argument('--max_output_length', type=int, default=64)
    argp.add_argument('--ignore_cache', action='store_true')

    training_args, args = argp.parse_args_into_dataclasses()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = load_raw_dataset(args.data_train)
    eval_dataset = load_raw_dataset(args.data_eval)

    def preprocess_fn(exs):
        inputs = exs['input_text']
        targets = exs['target_text']
        # inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding='max_length', truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_output_length, padding='max_length', truncation=True)

        labels['input_ids'] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']
        ]
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    with training_args.main_process_first(desc='data preprocessing'):
        train_dataset = train_dataset.map(
            preprocess_fn, batched=True, num_proc=1,
            remove_columns=['input_text', 'target_text'],
            load_from_cache_file=not args.ignore_cache
        )
        eval_dataset = eval_dataset.map(
            preprocess_fn, batched=True, num_proc=1,
            remove_columns=['input_text', 'target_text'],
            load_from_cache_file=not args.ignore_cache
        )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    results = trainer.train()
    trainer.save_model()

    print(results)
