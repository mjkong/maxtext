#!/bin/bash

cd $HOME

source .env/bin/activate

export KAGGLE_USERNAME=
export KAGGLE_KEY=
export DEBIAN_FRONTEND=noninteractive
export CHKPT_BUCKET=
export CKPT_ROOT_PATH=$HOME/gemma
export UNSCANNED_CKPT_PATH=`cat $HOME/conversion_checkpoint.log | tail -1 | awk '{print $6}'`
export TOKENIZER_PATH=assets/tokenizer.gemma
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=gemma-7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=-1
export ICI_TENSOR_PARALLELISM=1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=11

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python JetStream/benchmarks/benchmark_serving.py \
--tokenizer maxtext/assets/tokenizer.gemma \
--num-prompts 1000 \
--dataset sharegpt \
--dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--request-rate 5 \
--warmup-mode full