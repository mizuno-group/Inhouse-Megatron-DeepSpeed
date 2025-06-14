#!/bin/sh
#PBS -q regular-c
#PBS -l select=1
#PBS -W group_list=gg17
#PBS -o preprocess-2.out
#PBS -e preprocess-2.err

module purge
module load cmake
module load gcc

source /work/gg17/a97006/.c_bashrc

cd ~/env/env-c
source ./250/bin/activate

wandb login 65afaa936940cf3a198fba3da2d51b71b797b77e

set -e

BASE_DIR="/work/gg17/a97006/250519_modern_bert_0"
SCRIPT_PATH="${BASE_DIR}/Inhouse-Megatron-DeepSpeed/tools/preprocess_data.py"
VOCAB_FILE="${BASE_DIR}/tokenizer/vocab_150000.txt"

TOKENIZER_TYPE="BertWordPieceCase"
WORKERS=48
WANDB_PROJECT="med_preprocess"
DATE_TAG="250613" 

DATASETS="pubmed pmc fda_label nih_books"

for dataset_name in ${DATASETS}; do

    echo "=================================================="
    echo "Processing dataset: ${dataset_name}"
    echo "=================================================="

    input_file="${BASE_DIR}/json/${dataset_name}.jsonl"
    output_prefix_dir="${BASE_DIR}/preprocessed/${dataset_name}/pubmed_150000"
    output_prefix="${output_prefix_dir}/${dataset_name}"
    wandb_name="${DATE_TAG}-miyabi-${dataset_name}-150000"

    mkdir -p "${output_prefix_dir}"

    python "${SCRIPT_PATH}" \
        --input "${input_file}" \
        --output-prefix "${output_prefix}" \
        --vocab-file "${VOCAB_FILE}" \
        --tokenizer-type "${TOKENIZER_TYPE}" \
        --workers "${WORKERS}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-name "${wandb_name}"

    echo "Finished processing ${dataset_name}"
    echo ""

done

echo "All datasets have been processed."