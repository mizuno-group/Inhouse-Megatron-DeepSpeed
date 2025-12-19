#!/bin/sh
#PBS -q regular-c
#PBS -l select=1
#PBS -W group_list=gg17
#PBS -o preprocess_c4.out
#PBS -e preprocess_c4.err

# --- 環境設定 ---
module purge
module load cmake
module load gcc

source /work/gg17/a97006/.c_bashrc

cd ~/env/env-c
source ./250/bin/activate

pip install datasets
pip install huggingface_hub
pip install dask

wandb login 65afaa936940cf3a198fba3da2d51b71b797b77e

# コマンドが失敗した場合、直ちにスクリプトを終了する
set -e

# --- 基本設定 ---
BASE_DIR="/work/gg17/a97006/250519_modern_bert_0"
# ダウンロード用スクリプトのパス
DOWNLOAD_SCRIPT_PATH="${BASE_DIR}/Inhouse-Megatron-DeepSpeed/ohto-shell/preprocess/dl_c4.py"
# 前処理用スクリプトのパス
PREPROCESS_SCRIPT_PATH="${BASE_DIR}/Inhouse-Megatron-DeepSpeed/tools/preprocess_data_sw.py"
# Vocabファイルのパス
VOCAB_FILE="/work/gg17/a97006/250519_modern_bert_0/tokenizer/bert-large-uncased-vocab.txt"

TOKENIZER_TYPE="BertWordPieceCase"
WORKERS=64
WANDB_PROJECT="med_preprocess"
DATE_TAG="250911"
DATASET_NAME="c4"

# --- データセット固有の設定 ---
# ダウンロードしたjsonlファイルを保存するディレクトリ
# また、前処理済みデータの出力先ディレクトリでもある
OUTPUT_DIR="${BASE_DIR}/Inhouse-Megatron-DeepSpeed/dataset/${DATASET_NAME}_bert_large"
# ダウンロードされるjsonlファイルのフルパス
INPUT_JSONL_FILE="${OUTPUT_DIR}/${DATASET_NAME}.jsonl"
# 前処理後の出力ファイル名のプレフィックス
OUTPUT_PREFIX="${OUTPUT_DIR}/${DATASET_NAME}"

# --- WandB用の設定 ---
WANDB_NAME="${DATE_TAG}-miyabi-${DATASET_NAME}"

# --- 実行 ---
echo "=================================================="
echo "Starting preprocessing for dataset: ${DATASET_NAME}"
echo "=================================================="

# 出力ディレクトリを作成
mkdir -p "${OUTPUT_DIR}"

# --- ステップ1: C4データセットのダウンロード ---
# echo "Step 1: Downloading C4 dataset..."
# python "${DOWNLOAD_SCRIPT_PATH}" \
#     --save_dir "${OUTPUT_DIR}"
# echo "Download complete. Raw data saved in ${OUTPUT_DIR}"
# echo "--------------------------------------------------"


# --- ステップ2: Megatron-LM用のデータ前処理 ---
echo "Step 2: Running Megatron preprocessing..."
python "${PREPROCESS_SCRIPT_PATH}" \
    --input "/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/dataset/c4/c4.jsonl" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --vocab-file "${VOCAB_FILE}" \
    --tokenizer-type "${TOKENIZER_TYPE}" \
    --workers "${WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --seq-length 1021 \
    --sliding-window-stride 512 \

echo "Finished processing ${DATASET_NAME}"
echo "Preprocessed .bin and .idx files are located at: ${OUTPUT_PREFIX}"
echo "=================================================="
echo "All datasets have been processed."