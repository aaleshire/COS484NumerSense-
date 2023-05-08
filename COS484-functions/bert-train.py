from typing import Tuple
import argparse
import torch
from transformers import (
    BertTokenizerFast,
    LineByLineTextDataset,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel

def mask_tokens(inputs: torch.Tensor, tokenizer: AutoTokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: mask number words. """
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


# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument("--model_type", type=str, default="bert")
parser.add_argument("--model_name_or_path", type=str, default="princeton-nlp/unsup-simcse-roberta-base")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--train_data_file", type=str, required=True)
parser.add_argument("--per_gpu_train_batch_size", type=int, default=64)
parser.add_argument("--block_size", type=int, default=64)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--line_by_line", action="store_true")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")


if args.line_by_line:
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.train_data_file,
        block_size=args.block_size,
    )
else:
    raise NotImplementedError

model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")

# Train with custom mask_tokens function
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_gpu_train_batch_size=args.per_gpu_train_batch_size,
        logging_steps=args.logging_steps,
    ),
    train_dataset=train_dataset.map(
        lambda x: mask_tokens(x['input_ids'], tokenizer)),
)

trainer.train()

# Save the trained model to the output directory
trainer.save_model(args.output_dir)
