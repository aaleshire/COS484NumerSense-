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

### Fine-tune a MLM model 
```bash
mkdir saved_models

BERT-Large:
CUDA_VISIBLE_DEVICES=0 python src/finetune_mlm.py \
  --output_dir=saved_models/finetuned_bert_large --overwrite_output_dir \
  --model_type=bert \
  --model_name_or_path=bert-large-uncased \
  --do_train \
  --train_data_file=COS484-data/gkb_best_filtered.txt  \
  --per_gpu_train_batch_size 64 \
  --block_size 64 \
  --logging_steps 100 \
  --num_train_epochs 3 \
  --line_by_line --mlm 

python  src/mlm_predict.py \
        reload_bert:saved_models/finetuned_bert_large \
        COS484-data/validation.masked.removed.txt \
        COS484-results/bert-large.finetune.validation.results

RoBERTa-Large:
CUDA_VISIBLE_DEVICES=0 python src/finetune_mlm.py \
  --output_dir=saved_models/finetuned_roberta_large --overwrite_output_dir \
  --model_type=roberta \
  --model_name_or_path=roberta-large \
  --do_train \
  --train_data_file=COS484-data/gkb_best_filtered.txt  \
  --per_gpu_train_batch_size 64 \
  --block_size 64 \
  --logging_steps 100 \
  --num_train_epochs 3 \
  --line_by_line --mlm 

python  src/mlm_predict.py \
        reload_roberta:saved_models/finetuned_roberta_large \
        COS484-data/validation.masked.removed.txt \
        COS484-results/roberta-large.finetune.validation.results
```

### For DistilBERT Baselines and Finetuning (need Colab's GPU for it to be a reasonable amount of time though):
```bash
python COS484-functions/distilbert-code.py
```

## Evaluating
```bash
GPT-2:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/gpt.validation.results.jsonl

BERT-Base:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/bert-base.validation.results.jsonl

RoBERTa-Base:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/roberta-base.validation.results.jsonl

BERT-Large:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/bert-large.validation.results.jsonl

RoBERTa-Large:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/roberta-large.validation.results.jsonl

BERT-Large-Finetuned:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/bert-large.finetune.validation.results

RoBERTa-Large-Finetuned:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/roberta-large.finetune.validation.results

DistilBERT:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/distilbert.validation.results.jsonl

DistilBERT-Finetuned:
python  COS484-functions/evaluator.py COS484-data/validation.masked.tsv COS484-results/distilbert.finetune.validation.results.jsonl 
```