#!/bin/sh
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -W group_list=gd43
#PBS -o convert_new.out
#PBS -e convert_new.err

# --- 0. 環境設定 ---
echo "--- 環境を設定しています... ---"
module purge
module load cmake
module load gcc
module load cuda/12.6
module load cudnn/9.5.1.17
module load ompi-cuda/4.1.6-12.6

source /work/gg17/a97006/.g_bashrc
cd ~/env/llm-pyenv-3
source ./250/bin/activate

export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.6"
export LD_LIBRARY_PATH=/work/opt/local/aarch64/cores/jupyterlab/4.3.5/python/3.12.8/lib:\
$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset OMPI_MCA_mca_base_env_list

export PYTHONPATH="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed"
CHECKPOINT_DIR="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/a97006/project/bert_with_pile/checkpoint/250819_pubmed-0.149B-iters-2M-lr-8e-4-min-1e-5-wmup-35800-dcy-2M-sty-constant-gbs-1280-mbs-20-gpu-64-zero-0-mp-1-pp-1-nopp"
OUTPUT_PATH="/work/gg17/a97006/hf_models/med-bert-before-decay"

python3 /work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/ohto-shell/model_convert/zero_to_fp32.py $CHECKPOINT_DIR $OUTPUT_PATH --tag global_step70000


UNIQUE_NODES_FILE=$(mktemp)
if [ -z "$PBS_NODEFILE" ]; then
    echo "Error: PBS_NODEFILE is not set. Running on localhost."
    echo "$(hostname)" > $UNIQUE_NODES_FILE
    # Fallback for local testing if not in PBS environment
    num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    num_node=1
    # The original num_gpus calculation. This is kept for reference but is fragile.
    # It assumes ds_ssh can run on all nodes and sums up GPU counts.
    # The -2 is suspicious (might be correcting for header/footer lines from ds_ssh wrapper).
    # num_gpus_original_calc=$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
else
    sort -u $PBS_NODEFILE > $UNIQUE_NODES_FILE
fi

if [ -s $UNIQUE_NODES_FILE ]; then
    export MASTER_ADDR=$(head -n 1 $UNIQUE_NODES_FILE)
else
    export MASTER_ADDR=$(hostname) # Fallback for local/single node
fi
export MASTER_PORT=29500 # Choose a free port

# # 最終的なHugging Face形式の保存先
HF_MODEL_PATH="/work/gg17/a97006/hf_models/250820"
vocab_path="/work/gg17/a97006/250519_modern_bert_0/tokenizer/vocab_100000.txt"

python /work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/tools/convert_checkpoint/modern_bert_converter_bin.py \
    --load ${OUTPUT_PATH}/pytorch_model.bin \
    --save /work/gg17/a97006/before_decay \
    --num-layers 22 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --save-interval 1000 \
    --vocab-file ${vocab_path} \
    --tokenizer-type BertWordPieceCase \
    --target_vocab_size 100096 \

echo ""
echo "##############################################"
echo "### 全ての変換プロセスが正常に完了しました！ ###"
echo "##############################################"