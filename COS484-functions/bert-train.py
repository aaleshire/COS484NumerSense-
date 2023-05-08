import argparse
import os
import torch
from transformers import BertTokenizerFast, BertForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from typing import Tuple


def mask_tokens(inputs: torch.Tensor, tokenizer: BertTokenizerFast) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: mask numerical words. """
    num_list = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "no", "zero"]
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    labels = inputs.clone()
    masked_indices = []
    for input in inputs:
        current_masked_indices = []
        tokens = tokenizer.convert_ids_to_tokens(input)
        for token_id, token in enumerate(tokens):
            if token in num_list:
                current_masked_indices.append(True)
            else:
                current_masked_indices.append(False)
        masked_indices.append(current_masked_indices)

    masked_indices = torch.BoolTensor(masked_indices)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="ashraq/bert-random-weights")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--line_by_line", action="store_true")
    args = parser.parse_args()

    # Load the tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    model = BertForMaskedLM.from_pretrained(args.model_name_or_path)

    # Load the text file into a dataset
    if args.line_by_line:
        train_dataset = LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=args.train_data_file,
            block_size=args.block_size,
        )
    else:
        raise NotImplementedError

    # Define a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=1.0, mask_token=tokenizer.mask_token, mask_token_id=tokenizer.mask_token_id,
        masking_function=mask_tokens
    )

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.logging_steps,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)
