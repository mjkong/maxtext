## JetStream MaxText 테스트 on Google Cloud TPUs

> 참조문서: https://cloud.google.com/tpu/docs/tutorials/LLM/jetstream?hl=ko#convert_model_checkpoints

### [Step 1] Download sources
```bash
export KAGGLE_USERNAME=<Kaggle Username>
export KAGGLE_KEY=<Kaggle Key>
export DEBIAN_FRONTEND=noninteractive
export CHKPT_BUCKET=<Checkpoint 저장을 위한 버킷>
export CKPT_ROOT_PATH=$HOME/gemma

sudo apt update -y
sudo apt remove needrestart -y
sudo apt install -y pre-commit python3.10-venv python-is-python3

git clone -b summit24 https://github.com/mjkong/maxtext.git
git clone -b summit24 https://github.com/mjkong/JetStream.git
```

### [Step 2] Download Gemma checkpoint
```bash
mkdir -p $CKPT_ROOT_PATH
wget https://www.kaggle.com/api/v1/models/google/gemma/maxtext/7b/2/download --user=$KAGGLE_USERNAME --password=$KAGGLE_KEY --auth-no-challenge -O $CKPT_ROOT_PATH/download
tar -xf $CKPT_ROOT_PATH/download -C $CKPT_ROOT_PATH

gsutil mb $CHKPT_BUCKET
gsutil -m cp -r ${CKPT_ROOT_PATH}/7b ${CHKPT_BUCKET}
```

### [Step 3] Configure MaxText
```bash
cd ~
python -m venv .env
source .env/bin/activate

cd maxtext 
bash setup.sh

# For gemma-7b
bash ../JetStream/jetstream/tools/maxtext/model_ckpt_conversion.sh gemma 7b ${CHKPT_BUCKET}/7b
```

### [Step 4] Start server
위 명령  실행의 결과 로그에서 확인 가능한 UNSCANNED_CKPT_PATH 환경변수를 적용<br/>
예: export UNSCANNED_CKPT_PATH=<GCS bucket url>/gemma-7b_unscanned_chkpt_2024-06-26-03-04/checkpoints/0/items

```bash
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
```

### [Step 5] Reqest test query 
새로운 터미널일 열고 Python venv 에서 상위행적용했던 환경변수 적용 후 실행
```bash
python JetStream/jetstream/tools/requester.py --tokenizer maxtext/assets/tokenizer.gemma
```

### [Step 6] Run benchmark test
Step 5의 터미널에서 실행
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python JetStream/benchmarks/benchmark_serving.py \
--tokenizer maxtext/assets/tokenizer.gemma \
--num-prompts 1000 \
--dataset sharegpt \
--dataset-path ~/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-output-length 1024 \
--request-rate 5 \
--warmup-mode full
```