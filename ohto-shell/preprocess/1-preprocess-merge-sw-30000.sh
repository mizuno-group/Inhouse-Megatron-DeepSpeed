#!/bin/sh
#PBS -q regular-c
#PBS -l select=1
#PBS -W group_list=gd43
#PBS -o preprocess_merged-3.out
#PBS -e preprocess_merged-3.err

module purge
module load cmake
module load gcc

source /work/gg17/a97006/.c_bashrc
cd ~/env/env-c
source ./250/bin/activate

wandb login 65afaa936940cf3a198fba3da2d51b71b797b77e

set -e

BASE_DIR="/work/gg17/a97006/0-250519_modern_bert_0"
SCRIPT_PATH="${BASE_DIR}/Inhouse-Megatron-DeepSpeed/tools/preprocess_data_sw.py"
VOCAB_FILE="${BASE_DIR}/251004_tokenizer/vocab_30000.txt"
TOKENIZER_TYPE="BertWordPieceCase"
WORKERS=20
WANDB_PROJECT="med_preprocess"
DATE_TAG="251005_preprocess-3"

echo "=================================================="
echo "Step 1: Preprocessing the merged file..."
echo "=================================================="

OUTPUT_PREFIX_DIR="${BASE_DIR}/251004_preprocessed/4_merged/all_merged_30000-1024"
mkdir -p "${OUTPUT_PREFIX_DIR}"
OUTPUT_PREFIX="${OUTPUT_PREFIX_DIR}/4_merged"
WANDB_NAME="${DATE_TAG}"

MERGED_JSON_FILE="/work/gg17/a97006/0-250519_modern_bert_0/251001_json/251003_merged.jsonl"

python "${SCRIPT_PATH}" \
    --input "${MERGED_JSON_FILE}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --vocab-file "${VOCAB_FILE}" \
    --tokenizer-type "${TOKENIZER_TYPE}" \
    --workers "${WORKERS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --seq-length 1021 \
    --sliding-window-stride 512 \

echo ""
echo "All datasets have been processed into a single dataset."