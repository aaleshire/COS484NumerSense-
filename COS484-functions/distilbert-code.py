# make sure have happy transformer installed
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from typing import Optional, Any, Tuple
import torch
import json
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from transformers import FillMaskPipeline, AutoModelForMaskedLM, PretrainedConfig

from happytransformer.happy_transformer import HappyTransformer
from happytransformer.cuda_detect import detect_cuda_device_number
from happytransformer.adaptors import get_adaptor
from happytransformer.wp import ARGS_WP_TRAIN, ARGS_WP_EVAl, ARGS_WP_TEST
from happytransformer.happy_trainer import EvalResult
from happytransformer.fine_tuning_util import create_args_dataclass


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, words_to_mask=None, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.words_to_mask = words_to_mask or [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "zero", "no"
        ]
        self.words_to_mask_ids = [tokenizer.convert_tokens_to_ids(word) for word in self.words_to_mask]

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        masked_indices = torch.zeros_like(inputs).bool()

        for word_id in self.words_to_mask_ids:
            word_mask = inputs == word_id
            masked_indices |= word_mask

        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.bool()
            masked_indices &= ~special_tokens_mask

        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

"""
Fine-tuning for masked word prediction models.

Based on the tutorial found here:
https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
"""
from dataclasses import dataclass
# from transformers import DataCollatorForLanguageModeling
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from datasets import load_dataset
from happytransformer.fine_tuning_util import preprocess_concatenate
from happytransformer.wp.default_args import ARGS_WP_TRAIN, ARGS_WP_EVAl, ARGS_WP_TEST
import json


@dataclass
class WPTrainArgs:
    learning_rate: float = ARGS_WP_TRAIN["learning_rate"]
    num_train_epochs: int = ARGS_WP_TRAIN["num_train_epochs"]
    batch_size: int = ARGS_WP_TRAIN["batch_size"]
    weight_decay: float = ARGS_WP_TRAIN["weight_decay"]
    adam_beta1: float = ARGS_WP_TRAIN["adam_beta1"]
    adam_beta2: float = ARGS_WP_TRAIN["adam_beta2"]
    adam_epsilon: float = ARGS_WP_TRAIN["adam_epsilon"]
    max_grad_norm:  float = ARGS_WP_TRAIN["max_grad_norm"]
    save_preprocessed_data: bool = ARGS_WP_TRAIN["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_WP_TRAIN["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_WP_TRAIN["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_WP_TRAIN["load_preprocessed_data_path"]
    preprocessing_processes: int = ARGS_WP_TRAIN["preprocessing_processes"]
    mlm_probability: float = ARGS_WP_TRAIN["mlm_probability"]
    line_by_line: bool = ARGS_WP_TRAIN["line_by_line"]
    fp16: bool = ARGS_WP_TRAIN["fp16"]


@dataclass
class WPEvalArgs:
    batch_size: int = ARGS_WP_EVAl["batch_size"]
    save_preprocessed_data: bool = ARGS_WP_EVAl["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_WP_EVAl["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_WP_EVAl["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_WP_EVAl["load_preprocessed_data_path"]
    preprocessing_processes: int =ARGS_WP_EVAl["preprocessing_processes"]
    mlm_probability: float = ARGS_WP_EVAl["mlm_probability"]
    line_by_line: bool = ARGS_WP_EVAl["line_by_line"]



class WPTrainer(HappyTrainer):
    """
    Trainer class for HappyWordPrediction
    """
    def train(self, input_filepath, dataclass_args: WPTrainArgs):
        """
        :param input_filepath: A file path to a text file that contains nothing but training data
        :param dataclass_args: A WPTrainArgs() object
        :return: None
        """
        if not dataclass_args.load_preprocessed_data: # this hits
            self.logger.info("Preprocessing dataset...")

            dataset = load_dataset("text", data_files={"train": input_filepath})
            if dataclass_args.line_by_line:
                tokenized_dataset = self._preprocess_line_by_line(self.tokenizer, dataset, dataclass_args.preprocessing_processes)
            else: # this hits
                # so this is the line that matters THOMAS ABBY
                tokenized_dataset = preprocess_concatenate(self.tokenizer, dataset, dataclass_args.preprocessing_processes, True)
        else:
            self.logger.info("Loading dataset from %s...", dataclass_args.load_preprocessed_data_path)
            tokenized_dataset = load_dataset("json", data_files={"train": dataclass_args.load_preprocessed_data_path}, field='train')

        if dataclass_args.save_preprocessed_data:
            if dataclass_args.load_preprocessed_data:
                self.logger.warning("Both save_preprocessed_data and load_data are enabled,")

            self.logger.info("Saving training dataset to %s...", dataclass_args.save_preprocessed_data_path)

            self._generate_json(dataclass_args.save_preprocessed_data_path, tokenized_dataset["train"], "train")

        # changed this THOMAS ABBY
        # data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                       # mlm_probability=dataclass_args.mlm_probability)

        data_collator = CustomDataCollatorForLanguageModeling(tokenizer=happy_wp.tokenizer, mlm_probability=0.15)


        self.logger.info("Training...")

        self._run_train(tokenized_dataset['train'], dataclass_args, data_collator)


    def eval(self, input_filepath, dataclass_args: WPEvalArgs):
        """
        :param input_filepath: A file path to a text file that contains nothing but evaluating data
        :param dataclass_args: A WPEvalArgs() object
        :return: An EvalResult() object
        """
        dataset = load_dataset("text", data_files={"eval": input_filepath})

        if dataclass_args.line_by_line:
            tokenized_dataset = self._preprocess_line_by_line(self.tokenizer, dataset, dataclass_args.preprocessing_processes)
        else:
            tokenized_dataset = preprocess_concatenate(self.tokenizer, dataset, dataclass_args.preprocessing_processes, True)


        data_collator = CustomDataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=dataclass_args.mlm_probability)
        result = self._run_eval(tokenized_dataset['eval'], data_collator, dataclass_args)

        return EvalResult(loss=result["eval_loss"])


    def test(self, input_filepath, solve, args):
        raise NotImplementedError()


    def _preprocess_line_by_line(self, tokenizer, dataset, preprocessing_processes):
        """
        :param tokenizer: tokenizer for a transformer model
        :param datasets: a datasets.Dataset object
        :param preprocessing_processes: number of processes to use for pre-processing
        :return:
        """

        def tokenize_function(example):
            return tokenizer(example["text"],
                             add_special_tokens=True, truncation=True,)

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                          num_proc=preprocessing_processes,
                                          remove_columns=["text"])
        return tokenized_dataset


    def _generate_json(self, json_path, dataset, name):
        """
        :param json_path: A path to a json file that will be created/overwritten
        :param dataset: A list of dictionaries that contain the keys "attention_mask," "input_ids" and "labels"
        :param name: A string to specify if the written data is for "Train" or "Eval"
        :return: None
        """
        data = {}
        data[name] = []
        data = {
            name: [
                {
                    'attention_mask': case['attention_mask'],
                    'input_ids': case['input_ids'],
                }
                for case in dataset
            ]
        }

        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)

@dataclass
class WordPredictionResult:
    token: str
    score: float

class HappyWordPrediction(HappyTransformer):
    """
    A user facing class for text classification
    """
    def __init__(
            self, model_type: str = "DISTILBERT", model_name: str = "distilbert-base-uncased",
            load_path: str ="", use_auth_token: str = None, from_tf=False):


        self.adaptor = get_adaptor(model_type)

        if load_path != "":
            model = AutoModelForMaskedLM.from_pretrained(load_path, from_tf=from_tf)
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_name, use_auth_token=use_auth_token, from_tf=from_tf)

        super().__init__(model_type, model_name, model, load_path=load_path, use_auth_token=use_auth_token)

        device_number = detect_cuda_device_number()

        self._pipeline = FillMaskPipeline(model=self.model, tokenizer=self.tokenizer, device=device_number)

        self._trainer = WPTrainer(self.model, model_type, self.tokenizer, self._device, self.logger)

    def predict_mask(self, text: str, targets: Optional[List[str]] = None, top_k: int = 1) -> List[WordPredictionResult]:
        """
        Predict [MASK] tokens in a string.
        targets limit possible guesses if supplied.
        top_k describes number of targets to return*
        *top_k does not apply if targets is supplied
        """
        if not isinstance(text, str):
            raise ValueError('the "text" argument must be a single string')

        text_for_pipeline = self.adaptor.preprocess_mask_text(text)
        answers = self._pipeline(
            text_for_pipeline, 
            targets=targets, top_k=top_k
        )

        fix_token = self.adaptor.postprocess_mask_prediction_token
        return [
            WordPredictionResult(
                token=fix_token(answer["token_str"]), 
                score=answer["score"]
            )
            for answer in answers
        ]

    def train(self, input_filepath, args=ARGS_WP_TRAIN):
        if type(args) == dict:
            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_WP_TRAIN,
                                                         input_dic_args=args,
                                                         method_dataclass_args=WPTrainArgs)
        elif type(args) == WPTrainArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a WPTrainArgs object or a dictionary")

        # this is the line that matters THOMAS ABBY
        self._trainer.train(input_filepath=input_filepath, dataclass_args=method_dataclass_args)

    def eval(self, input_filepath, args=ARGS_WP_EVAl) -> EvalResult:
        if type(args) == dict:

            method_dataclass_args = create_args_dataclass(default_dic_args=ARGS_WP_EVAl,
                                                         input_dic_args=args,
                                                         method_dataclass_args=WPEvalArgs)
        elif type(args) == WPEvalArgs:
            method_dataclass_args = args
        else:
            raise ValueError("Invalid args type. Use a ARGS_WP_EVAl object or a dictionary")

        return self._trainer.eval(input_filepath=input_filepath, dataclass_args=method_dataclass_args)


    def test(self, input_filepath, args=ARGS_WP_TEST):
        raise NotImplementedError("test() is currently not available")


def process_file(input_file, output_file):

    with open(input_file, "r") as file:
        lines = file.readlines()      

    with open(output_file, "w") as out_file:
        count = 0
        for line in lines:
            if (count % 20 == 0): print(count)
            count+=1
            prompt = line.replace("<mask>", "[MASK]")
            result = happy_wp.predict_mask(prompt, targets=targets, top_k=top_k)
            formatted_result = {
                "probe": line.strip(),
                "result_list": [{"word": r.token, "score": r.score} for r in result]
            }
            out_file.write(json.dumps(formatted_result) + "\n")

# THE ACTUAL WORK #
happy_wp = HappyWordPrediction()
targets = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "no"]
top_k = 12

# getting the base
input_file = "COS484-data/validation.masked.removed.txt"
output_file = "COS484-results/distilbert.validation.results.jsonl"
process_file(input_file, output_file)

# training
happy_wp.train("COS484-data/gkb_best_filtered.txt")

# getting the finetuned
input_file = "COS484-data/validation.masked.removed.txt"
output_file = "COS484-results/distilbert.validation.results.jsonl"
process_file(input_file, output_file)