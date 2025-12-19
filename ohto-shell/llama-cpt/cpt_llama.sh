#!/bin/sh
#PBS -q short-g
#PBS -l select=1
#PBS -W group_list=gg17
#PBS -o test.out
#PBS -e test.err

# === 1. 環境とパラメータの設定 ===
# --- HPC環境設定 ---
module purge
module load cmake
module load gcc
module load cuda/12.6
module load cudnn/9.5.1.17
module load ompi-cuda/4.1.6-12.6

# --- Python環境設定 ---
source /work/gg17/a97006/.g_bashrc
pyenv local 3.12.4
cd ~/env/llm-pyenv-3
source ./250/bin/activate

export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.6"
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset OMPI_MCA_mca_base_env_list

# --- パス設定 ---
# DeepSpeed設定ファイル
DS_CONFIG=./ds_config_llama_finetune.json
# データセットのパス
DATASET_PATH=./examples_deepspeed/finetune_hf_llama/alpaca_data.json
# 変換元のHugging Faceモデルのパス
HF_LLAMA_PATH=/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/model/llama-3.1-8b-hf
# 変換後・学習済みモデルを保存するMegatron-DS形式のパス
MEGA_DS_LLAMA_PATH=/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/model/"llama-3.1-8b-mega-ds-T${TP}P${PP}"
# 最終的なHugging Face形式の出力先
FINAL_HF_OUTPUT_PATH=${MEGA_DS_LLAMA_PATH}-hf-out

# --- モデル＆学習設定 ---
TP=2
PP=2

# === 2. 実行コマンドの事前定義 ===
# --- 変換コマンドの定義 ---
CONVERTER_SCRIPT="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/tools/hf2megads_weight_converter.py"

# 変換スクリプトのパス
CONVERTER_SCRIPT="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/tools/hf2megads_weight_converter.py"

# パラメータ設定
TP=2
PP=2
HF_LLAMA_PATH=/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/model/llama-3.1-8b-hf
MEGA_DS_LLAMA_PATH=/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/model/"llama-3.1-8b-mega-ds-T${TP}P${PP}"

# GPUとノード情報の動的取得
UNIQUE_NODES_FILE=$(mktemp)
if [ -z "$PBS_NODEFILE" ]; then
  echo "Error: PBS_NODEFILE is not set. Assuming single-node."
  hostname > $UNIQUE_NODES_FILE
else
  sort -u $PBS_NODEFILE > $UNIQUE_NODES_FILE
fi
num_node=$(wc -l < $UNIQUE_NODES_FILE)
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# mpirun / torchrun の設定
export MASTER_ADDR=$(head -n 1 $UNIQUE_NODES_FILE)
export MASTER_PORT=29500
HOST_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
MPIRUN_OPTIONS="--hostfile ${UNIQUE_NODES_FILE} -np ${num_node} -npernode 1 --map-by node -x CUDA_HOME -x LD_LIBRARY_PATH -x PATH"


# 実行コマンド
python ${CONVERTER_SCRIPT} \
  --load-mode auto \
  --save ${MEGA_DS_LLAMA_PATH} \
  --hf-ckpt-dir ${HF_LLAMA_PATH} \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --num-layers 32 \
  --hidden-size 4096 \
  --ffn-hidden-size 14336 \
  --num-attention-heads 32 \
  --num-key-value-heads 8 \
  --max-position-embeddings 8192 \
  --seq-length 8192 \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --tokenizer-type HFTokenizer \
  --tokenizer-model ${HF_LLAMA_PATH} \
  --bf16 \
  --use-rotary-position-embeddings \
  --swiglu \
  --normalization rmsnorm \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --train-iters 100 \
  --lr 1.0e-5 \
  --save-interval 100

# --- ファインチューニングコマンドの定義 ---
FINETUNE_SCRIPT="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/finetune_llama.py"

# DeepSpeed設定ファイルの生成
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": 0
  },
  "bf16": {
    "enabled": true
  }
}
EOT

# mpirun/torchrunで実行されるコマンド引数
finetune_args=" \
--load ${MEGA_DS_LLAMA_PATH} \
--save ${MEGA_DS_LLAMA_PATH} \
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size ${PP} \
--num-layers ${NUM_LAYERS} \
--hidden-size ${HIDDEN_SIZE} \
--ffn-hidden-size ${FFN_HIDDEN_SIZE} \
--num-attention-heads ${NUM_HEADS} \
--num-key-value-heads ${NUM_KV_HEADS} \
--seq-length ${SEQ_LENGTH} \
--max-position-embeddings ${SEQ_LENGTH} \
--micro-batch-size ${MICRO_BATCH_SIZE} \
--global-batch-size ${GLOBAL_BATCH_SIZE} \
--train-iters 200 \
--lr 2e-5 \
--lr-decay-style cosine \
--weight-decay 0.1 \
--clip-grad 1.0 \
--bf16 \
--finetune \
--log-interval 10 \
--eval-interval 100 \
--eval-iters 10 \
--save-interval 500 \
--data-path ${DATASET_PATH} \
--split 98,2,0 \
--tokenizer-type HFTokenizer \
--tokenizer-model ${HF_LLAMA_PATH} \
--use-rotary-position-embeddings \
--swiglu \
--normalization rmsnorm \
--disable-bias-linear \
--no-query-key-layer-scaling \
--untie-embeddings-and-output-weights \
--attention-dropout 0 \
--hidden-dropout 0 \
--deepspeed \
--deepspeed_config ${DS_CONFIG} \
--zero-stage 0"


#################################################################
#                   ここから実行ワークフロー                      #
#################################################################

echo "========== START: E2E Llama Fine-tuning Workflow =========="

# 手順3: ファインチューニングの実行
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256
# Llama 3.1 8Bのアーキテクチャ
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_LAYERS=32
NUM_HEADS=32
NUM_KV_HEADS=8 # GQAのために追加
SEQ_LENGTH=8192 # メモリに合わせて調整

echo -e "\n\n--- STEP 2: Starting fine-tuning with mpirun and torchrun ---"
# GPUとノード情報の動的取得
UNIQUE_NODES_FILE=$(mktemp)
if [ -z "$PBS_NODEFILE" ]; then
  echo "Error: PBS_NODEFILE is not set. Assuming single-node."
  hostname > $UNIQUE_NODES_FILE
else
  sort -u $PBS_NODEFILE > $UNIQUE_NODES_FILE
fi
num_node=$(wc -l < $UNIQUE_NODES_FILE)
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# mpirun / torchrun の設定
export MASTER_ADDR=$(head -n 1 $UNIQUE_NODES_FILE)
export MASTER_PORT=29500
HOST_IP=$(getent hosts $MASTER_ADDR | awk '{print $1}')
MPIRUN_OPTIONS="--hostfile ${UNIQUE_NODES_FILE} -np ${num_node} -npernode 1 --map-by node -x CUDA_HOME -x LD_LIBRARY_PATH -x PATH"

# 実行コマンドの組み立て
TORCHRUN_CMD="torchrun \
  --nnodes ${num_node} \
  --nproc_per_node ${num_gpus_pernode} \
  --rdzv_id ${PBS_JOBID:-$RANDOM} \
  --rdzv_backend c10d \
  --rdzv_endpoint ${HOST_IP}:${MASTER_PORT} \
  ${FINETUNE_SCRIPT} \
  ${finetune_args}"

# 実行
mpirun ${MPIRUN_OPTIONS} ${TORCHRUN_CMD}
rm -f $UNIQUE_NODES_FILE


# 手順4: ファインチューニング済みモデルをMegatron-DS形式からHugging Face形式へ変換
echo -e "\n\n--- STEP 3: Converting fine-tuned model back to Hugging Face format ---"
eval "$covert_mds2hf_cmd"


echo -e "\n\n========== E2E Workflow Finished =========="