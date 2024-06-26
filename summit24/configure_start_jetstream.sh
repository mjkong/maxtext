#!/bin/bash

### [Step 1] Download sources
export KAGGLE_USERNAME=
export KAGGLE_KEY=
export DEBIAN_FRONTEND=noninteractive
export CHKPT_BUCKET=
export CKPT_ROOT_PATH=$HOME/gemma

gsutil rm -far $CHKPT_BUCKET
gsutil rm -far gs://$USER-maxtext 
gsutil rm -far gs://$USER-maxtext-dataset 
gsutil rm -far gs://$USER-runner-maxtext-logs

sudo apt update -y
sudo apt remove needrestart -y
sudo apt install -y pre-commit python3.10-venv python-is-python3

git clone -b summit24 https://github.com/mjkong/maxtext.git
git clone -b summit24 https://github.com/mjkong/JetStream.git

### [Step 2] Download Gemma checkpoint
mkdir -p $CKPT_ROOT_PATH
wget https://www.kaggle.com/api/v1/models/google/gemma/maxtext/7b/2/download --user=$KAGGLE_USERNAME --password=$KAGGLE_KEY --auth-no-challenge -O $CKPT_ROOT_PATH/download
tar -xf $CKPT_ROOT_PATH/download -C $CKPT_ROOT_PATH

gsutil mb $CHKPT_BUCKET
gsutil -m cp -r ${CKPT_ROOT_PATH}/7b ${CHKPT_BUCKET}

### [Step 3] Configure MaxText
cd ~
python -m venv .env
source .env/bin/activate

cd maxtext 
bash setup.sh

# For gemma-7b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh gemma 7b ${CHKPT_BUCKET}/7b >> $HOME/conversion_checkpoint.log

export UNSCANNED_CKPT_PATH=`cat $HOME/conversion_checkpoint.log | tail -2 | awk 'NR==1 {print $6}'`
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

python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE}