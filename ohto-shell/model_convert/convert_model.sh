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

# --- 1. パスと設定 ---
echo "--- パスと設定を定義します ---"
# 共通のPYTHONPATH
export PYTHONPATH="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed"

# # DeepSpeedチェックポイントのパス
DS_CHECKPOINT_PATH="/work/gg17/a97006/hf_models/med-bert-100000-470000/pytorch_model.bin"


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
    --load ${DS_CHECKPOINT_PATH} \
    --save /work/gg17/a97006/hf_models_250818_full \
    --num-layers 22 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --bf16 \
    --save-interval 1000 \
    --vocab-file ${vocab_path} \
    --tokenizer-type BertWordPieceCase \
    --target_vocab_size 100096 \

echo ""
echo "##############################################"
echo "### 全ての変換プロセスが正常に完了しました！ ###"
echo "##############################################"
echo "最終的なHugging Faceモデルは ${HF_MODEL_PATH} に保存されています。"


# # --- 自動変換スクリプトを実行 ---
# # 全ての実験フォルダが格納されている親ディレクトリ
# CHECKPOINT_BASE_DIR="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/a97006/project/bert_with_pile/checkpoint"

# # 変換後のHugging Faceモデルを保存する基本ディレクトリ
# HF_OUTPUT_BASE_DIR="/work/gg17/a97006/hf_models_converted"

# # 前回修正したPython変換スクリプトへのフルパス
# CONVERTER_SCRIPT="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/tools/convert_checkpoint/modern_bert_converter.py"

# # Python仮想環境を有効化（PBSスクリプト内でも実行しますが、念のため）
# source "/work/gg17/a97006/env/llm-pyenv-3/250/bin/activate"

# # Megatron-DeepSpeedライブラリへのPYTHONPATH
# export PYTHONPATH="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed"

# get_padded_vocab_size() {
#     local base_size=$1
#     local padding_multiple=128
#     echo $(((base_size + padding_multiple - 1) / padding_multiple * padding_multiple))
# }

# run_conversion() {
#     local parent_dir=$1
#     local step=$2
#     local vocab_base=$3 # 例: 30000
#     local model_name=$4 # 例: med-bert-full-30000-all-merged...

#     echo ""
#     echo "======================================================================"
#     echo ">> 変換を開始します:"
#     echo ">>   モデル: ${model_name}"
#     echo ">>   ステップ: ${step}"
#     echo ">>   語彙サイズ (ベース): ${vocab_base}"
#     echo "======================================================================"
    
#     # --- パスと引数の動的生成 ---
#     # 変換元のチェックポイントファイルへのフルパス
#     # ファイル名は環境に合わせて 'mp_rank_00_model_states.pt' または 'model_optim_rng.pt' に変更してください
#     local ds_checkpoint_path="${parent_dir}/global_step${step}/mp_rank_00_model_states.pt"

#     # 変換後のモデルの保存先パス
#     local hf_model_path="${HF_OUTPUT_BASE_DIR}/${model_name}-step${step}-hf"

#     # vocab_baseから実際の語彙サイズを計算（例: 30000 -> 30080）
#     # このルールはモデルの学習設定に合わせてください
#     local target_vocab_size=$(get_padded_vocab_size ${vocab_base})
#     local vocab_path="/work/gg17/a97006/250519_modern_bert_0/tokenizer/vocab_${vocab_base}.txt"
#     # --- 事前チェック ---
#     if [ ! -f "${ds_checkpoint_path}" ]; then
#         echo "   [エラー] チェックポイントファイルが見つかりません: ${ds_checkpoint_path}"
#         return 1
#     fi
    
#     mkdir -p "${hf_model_path}"

#     # --- Python変換スクリプトの実行 ---
#     # deepspeedランチャーを使うことで、分散環境の初期化を正しく行います
#     deepspeed --num_gpus=1 "${CONVERTER_SCRIPT}" \
#         --load "${ds_checkpoint_path}" \
#         --save "${hf_model_path}" \
#         --target_vocab_size ${target_vocab_size} \
#         --vocab-file ${vocab_path} \
#         --num-layers 22 \
#         --hidden-size 768 \
#         --num-attention-heads 12 \
#         --seq-length 1024 \
#         --save-interval 1000 \
#         --max-position-embeddings 1024 \
#         --tokenizer-type BertWordPieceCase \
#         --fp16 \
#         --tensor-model-parallel-size 1 \
#         --pipeline-model-parallel-size 1 \
#         --micro-batch-size 1 \
#         --global-batch-size 1 \

#     if [ $? -eq 0 ]; then
#         echo ">> 変換成功！ モデルは ${hf_model_path} に保存されました。"
#     else
#         echo ">> [エラー] 変換中にエラーが発生しました。"
#     fi
#     echo "======================================================================"
# }


# # --- 2. メイン処理 ---
# # 指定されたベースディレクトリ以下の全ての実験フォルダをループ処理
# echo "--- 自動変換処理を開始します ---"

# for parent_dir in ${CHECKPOINT_BASE_DIR}/med-bert-full-*; do
#     # ディレクトリでなければスキップ
#     [ -d "${parent_dir}" ] || continue

#     model_name=$(basename "${parent_dir}")
#     echo ""
#     echo "----------------------------------------------------------------------"
#     echo "実験フォルダを処理中: ${model_name}"
    
#     # フォルダ名から語彙サイズ（ベース）を抽出 (例: ...-30000-... -> 30000)
#     vocab_base=$(echo "${model_name}" | grep -o -E '[0-9]+' | head -n 1)
#     if [ -z "${vocab_base}" ]; then
#         echo "   [警告] 語彙サイズをフォルダ名から抽出できませんでした。スキップします。"
#         continue
#     fi
#     echo "   検出された語彙サイズ (ベース): ${vocab_base}"

#     # global_stepXXXX のステップ番号を抽出し、数値順にソート
#     steps=$(find "${parent_dir}" -maxdepth 1 -type d -name "global_step*" | sed 's/.*global_step//' | sort -n)
#     if [ -z "${steps}" ]; then
#         echo "   [警告] global_stepフォルダが見つかりませんでした。スキップします。"
#         continue
#     fi

#     # 最小ステップと最大ステップを取得
#     min_step=$(echo "${steps}" | head -n 1)
#     max_step=$(echo "${steps}" | tail -n 1)
    
#     echo "   最小ステップ: ${min_step}"
#     echo "   最大ステップ: ${max_step}"

#     # 最小ステップのチェックポイントを変換
#     run_conversion "${parent_dir}" "${min_step}" "${vocab_base}" "${model_name}"

#     # 最小と最大が異なる場合のみ、最大ステップのチェックポイントを変換
#     if [ "${min_step}" != "${max_step}" ]; then
#         run_conversion "${parent_dir}" "${max_step}" "${vocab_base}" "${model_name}"
#     fi
# done

# echo ""
# echo "##############################################"
# echo "### 全ての自動変換プロセスが完了しました！ ###"
# echo "##############################################"

# echo ""
# echo "##############################################"
# echo "### PBSジョブが正常に完了しました！ ###"
# echo "##############################################"