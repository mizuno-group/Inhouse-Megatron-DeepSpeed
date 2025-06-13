#!/bin/sh
#PBS -q debug-c
#PBS -l select=1
#PBS -W group_list=gg17
#PBS -o preprocess.out
#PBS -e preprocess.err

module purge
module load cmake
module load gcc

source /work/gg17/a97006/.c_bashrc

cd ~/env/env-c
source ./250/bin/activate

pip install nltk
wandb login 65afaa936940cf3a198fba3da2d51b71b797b77e

python /work/gg17/a97006/250519_modern_bert_0/Megatron-DeepSpeed/tools/preprocess_data.py \
    --input /work/gg17/a97006/250519_modern_bert_0/json/pubmed.jsonl \
    --output-prefix /work/gg17/a97006/250519_modern_bert_0/preprocessed/pubmed/pubmed_30000/pubmed \
    --vocab-file /work/gg17/a97006/250519_modern_bert_0/tokenizer/vocab_30000.txt \
    --tokenizer-type BertWordPieceCase \
    --workers 48 \
    --wandb-project med_preprocess \
    --wandb-name 250613-miyabi-pubmed-30000 \