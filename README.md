# NumerSense: Probing Numerical Commonsense Knowledge of BERTs


Project website: https://inklab.usc.edu/NumerSense/

Code & Data for EMNLP 2020 paper:

```bibtex
@inproceedings{lin2020numersense,
  title={Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-trained Language Models},
  author={Bill Yuchen Lin and Seyeon Lee and Rahul Khanna and Xiang Ren}, 
  booktitle={Proceedings of EMNLP},
  year={2020},
  note={to appear}
}
```

## Installation 
```bash
/**************************************/
            INSTALL CONDA
/**************************************/
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh
bash ./Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -f -p /usr/local/

conda install --channel defaults conda python=3.7 -y
conda update -n base -c defaults conda -y
conda create -n numersense python=3.7 -y


/**************************************/
        CONDA START ENVIRONMENT 
/**************************************/
sudo ln -s /opt/conda/root/etc/profile.d/conda.sh /etc/profile.d/conda.s 
eval "$(/usr/local/condabin/conda shell.bash hook)"
conda init bash
conda activate numersense


/**************************************/
            CLONE INTO REPO 
/**************************************/
git clone https://github.com/aaleshire/COS484NumerSense-
cd COS484NumerSense-
git checkout redo 


/**************************************/
          INSTALLATIONS NEEDED 
/**************************************/
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch -y

pip install transformers==3.3.1
pip install --editable happy-transformer
pip install happy-transformer
pip install tensorboardX


/**************************************/
        GITHUB CONFIGURATIONS 
/**************************************/
git config --global user.email "aleshire@princeton.edu"
git config --global user.name "Abby Aleshire"
```

## Probing Experiments 

```bash
GPT-2:
python  src/gpt_predict.py gpt \
        COS484-data/validation.masked.removed.txt \
        COS484-results/gpt.validation.results.jsonl 

BERT-Base:
python  src/mlm_predict.py bert-base \
        COS484-data/validation.masked.removed.txt \
        COS484-results/bert-base.validation.results.jsonl

RoBERTa-Base:
python  src/mlm_predict.py roberta-base \
        COS484-data/validation.masked.removed.txt \
        COS484-results/roberta-base.validation.results.jsonl 

BERT-Large:
python  src/mlm_predict.py bert-large \
        COS484-data/validation.masked.removed.txt \
        COS484-results/bert-large.validation.results.jsonl 

RoBERTa-Large:
python  src/mlm_predict.py roberta-large \
        COS484-data/validation.masked.removed.txt \
        COS484-results/roberta-large.validation.results.jsonl 
```

## Evaluating 

```bash
python  COS484-functions/evaluator.py 
        COS484-data/validation.masked.tsv 
        COS484-results/gpt.validation.results.jsonl
```
repeat for COS484-results/bert-base.validation.results.jsonl, COS484-results/roberta-base.validation.results.jsonl, COS484-results/bert-large.validation.results.jsonl, and COS484-results/roberta-large.validation.results.jsonl


### Fine-tune a MLM model 
```bash
mkdir saved_models
CUDA_VISIBLE_DEVICES=0 python src/finetune_mlm.py \
  --output_dir=saved_models/finetuned_bert_large --overwrite_output_dir \
  --model_type=bert \
  --model_name_or_path=bert-large-uncased \
  --do_train \
  --train_data_file=data/gkb_best_filtered.txt  \
  --do_eval \
  --eval_data_file=data/wiki_complete.txt \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --block_size 64 \
  --logging_steps 100 \
  --num_train_epochs 3 \
  --line_by_line --mlm 
```

```bash 
python src/mlm_infer.py \
        reload_bert:saved_models/finetuned_bert_large \
        data/test.core.masked.txt \
        results/test.core.output.jsonl
```

## Evaluation on Test Set

To evaluate your model's ability on NumerSense's official test sets,
please submit a prediction file to *yuchen.lin@usc.edu*, which should contain a json line for each probe example. And a json line should follow the format in the below code snippet. You can also check the example, `results/bert-base.test.core.output.jsonl` , which is the predictions of BERT-base on core set.
The `score` key is optional.
When submitting your predictions, please submit both `core` and `all` results, and inform us whether you have used the training data for fine-tuning. Thanks!
The evaluation script we will use is `src/evaluator.py`.
 ```json
{
  "probe": "a bird has <mask> legs.",
  "result_list": [
    {
      "word": "four",
      "score": 0.23623309
    },
    {
      "word": "two",
      "score": 0.21001829
    },
    {
      "word": "three",
      "score": 0.1258428
    },
    {
      "word": "no",
      "score": 0.0688955
    },
    {
      "word": "six",
      "score": 0.0639159
    },
    {
      "word": "five",
      "score": 0.061465383
    },
    {
      "word": "eight",
      "score": 0.038915534
    },
    {
      "word": "seven",
      "score": 0.014524153
    },
    {
      "word": "ten",
      "score": 0.010337788
    },
    {
      "word": "nine",
      "score": 0.005654324
    },
    {
      "word": "one",
      "score": 1.3131318E-4
    },
    {
      "word": "zero",
      "score": 1.10984496E-4
    }
  ]
}
 ```
