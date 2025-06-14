#!/bin/sh
#PBS -q short-g
#PBS -l select=2
#PBS -W group_list=gg17
#PBS -o test.out
#PBS -e test.err

module purge
module load cmake
module load gcc
module load cuda/12.6
module load cudnn/9.5.1.17
module load ompi-cuda/4.1.6-12.6

source /work/gg17/a97006/.g_bashrc
pyenv install 3.12.4 # This might take time; consider preparing an env beforehand
pyenv local 3.12.4

cd ~/env/llm-pyenv-3
source ./250/bin/activate

dir='/work/gg17/a97006/250519_modern_bert_0/Megatron-DeepSpeed/examples_deepspeed/bert_with_pile'
wandb login 65afaa936940cf3a198fba3da2d51b71b797b77e # Consider using environment variable WANDB_API_KEY
###############################################################################
seq_len=1024
global_batch_size=16
lr=1e-4
min_lr=1e-5

## BERT 110M (same config as original BERT-Base model)
model_size=0.11
num_layers=12
hidden_size=768
num_attn_heads=12
init_std=0.02
## BERT 336M (same config as original BERT-Large model)
# model_size=0.336
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# init_std=0.02
############################################################################### Training duration configs
train_iters_in_million=2
train_iters=$((${train_iters_in_million} * 10000)) # 2 * 10000 = 20000

###############################################################################
### lr configs
lr_warmup_iters=100 # これが lr_warmup_steps に対応
lr_decay_iters_in_million=${train_iters_in_million} # 2
lr_decay_iters=$((${lr_decay_iters_in_million} * 10000)) # 2 * 10000 = 20000
lr_decay_style="linear"
####################################################
### Parallelism configs
mp_size=1
pp_size=1
no_pp="true"
zero_stage=0

### GPU and Node calculation
# Get unique node names from PBS_NODEFILE
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

num_node=$(cat $UNIQUE_NODES_FILE | wc -l)
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) # GPUs on this specific node
num_gpus=$(( ${num_node} * ${num_gpus_pernode} )) # Total GPUs based on node count and GPUs per node

echo "INFO: Number of nodes: ${num_node}"
echo "INFO: GPUs per node (this node): ${num_gpus_pernode}"
echo "INFO: Total GPUs for training (world size): ${num_gpus}"
# echo "INFO: Original num_gpus calculation (for reference): ${num_gpus_original_calc}"


## Data parallel size.
# dp_size will be effectively the world_size for pure DP
dp_size=$(( ${num_gpus} / (${pp_size} * ${mp_size}) ))

## Micro batch size per GPU
if [ ${dp_size} -eq 0 ]; then
    echo "ERROR: dp_size is 0. Cannot divide by zero."
    exit 1
fi
batch_size=$(( ${global_batch_size} / ${dp_size} ))
if [ ${batch_size} -eq 0 ]; then
    echo "Warning: Calculated micro batch size is 0. Setting to 1. Check global_batch_size and dp_size."
    batch_size=1
fi
###############################################################################
### Misc configs
log_interval=1000
eval_iters=10
eval_interval=100
num_save=100
save_interval=$((${train_iters} / ${num_save}))
activation_checkpoint="false"
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}" # This will be the hostname of the node running this script (master PBS job)

jobname="bert-pubmed-test"

# BLEND DATASET
pubmed_path="/work/gg17/a97006/250519_modern_bert_0/preprocessed/pubmed/pubmed_30000/pubmed_text_document"
weight_pubmed=1.0
pmc_path="/work/gg17/a97006/250519_modern_bert_0/preprocessed/pmc/pubmed_30000/pmc_text_document"
weight_pmc=0.0
fda_label_path="/work/gg17/a97006/250519_modern_bert_0/preprocessed/fda_label/pubmed_30000/fda_label_text_document"
weight_fda_label=0.0
nih_books_path="/work/gg17/a97006/250519_modern_bert_0/preprocessed/nih_books/pubmed_30000/nih_books_text_document"
weight_nih_books=0.0
# Combine the datasets into a single data path
data_path="${weight_pubmed} ${pubmed_path} ${weight_pmc} ${pmc_path} ${weight_fda_label} ${fda_label_path} ${weight_nih_books} ${nih_books_path}"

vocab_path="/work/gg17/a97006/250519_modern_bert_0/tokenizer/vocab_30000.txt"

num_workers=4

jobname="${jobname}-${model_size}B-iters-${train_iters_in_million}M"
jobname="${jobname}-lr-${lr}-min-${min_lr}-wmup-${lr_warmup_iters}-dcy-${lr_decay_iters_in_million}M-sty-${lr_decay_style}"
jobname="${jobname}-gbs-${global_batch_size}-mbs-${batch_size}-gpu-${num_gpus}-zero-${zero_stage}-mp-${mp_size}-pp-${pp_size}"
if [ "${no_pp}" = "true" ]; then
    jobname="${jobname}-nopp"
fi

username=$(whoami)
output_home="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/${username}/project/bert_with_pile"
# This host check might not be relevant
if [[ "$host" == *"webxt"* ]]; then
    output_home="/blob/users/${username}/project/bert_with_pile"
fi
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
tensorboard_dir="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/users/${username}/project/bert_with_pile/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}" # host here refers to the master job submission host
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --data-path ${data_path} \
    --num-workers 128 \
    --data-impl mmap"

megatron_options=" \
    --bert-no-binary-head \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std ${init_std} \
    --tensor-model-parallel-size ${mp_size} \
    --lr-decay-iters ${lr_decay_iters} \
    --lr-warmup-iters ${lr_warmup_iters} \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-iters ${train_iters} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-workers ${num_workers} \
    --fp16 \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path} \
    --wandb-project deepspeed-megatron \
    --wandb-exp-name test_2gpus \
    --wandb-save-dir /work/gg17/a97006/250519_modern_bert_0/users/a97006/project/bert_with_pile"

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

template_json="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/examples_deepspeed/bert_with_pile/ds_config_bert_TEMPLATE.json"
config_json="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/examples_deepspeed/bert_with_pile/ds_config_bert_bsz${global_batch_size}_mbsz${batch_size}_log${log_interval}_zero${zero_stage}.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
fi

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}" # This is for DeepSpeed, not Megatron's PP

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel" # This flag might be specific to how pretrain_bert.py handles it
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

## Checkpoint iteration handling (run by the master PBS script instance)
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0

# This loop assumes nodes are named worker-0, worker-1, etc.
# And that ds_ssh is configured to reach these nodes.
# If your nodes in PBS_NODEFILE are different, this needs adjustment or should use $PBS_NODEFILE.
# For simplicity, if ds_ssh works by reading PBS_NODEFILE or similar, this might be okay.
# However, usually only rank 0 would perform such a check and broadcast.
# This existing logic tries to check on all nodes. Let's keep it if it worked for you.
# If ds_ssh is not available or nodes are not worker-N, this will fail.
# A safer approach is to let rank 0 process (from mpirun) handle this if possible within the Python script.
if [ "$num_node" -gt 0 ]; then # Only attempt if nodes are identified
    echo "Attempting to find latest checkpoint iteration..."
    # This original loop assumes nodes are worker-0, worker-1...
    # for (( node_idx = 0; node_idx < num_node; node_idx++ )) 
    # do
    #    current_host_to_check="worker-${node_idx}" # This is a potential point of failure if hostnames differ
    #    if $(ssh -q ${current_host_to_check} "test -f \"$iteration_file\""); then
    #        local_iteration=$(ssh -q ${current_host_to_check} cat $iteration_file)
    #        if [[ "${local_iteration}" =~ ^[0-9]+$ ]] && [[ "${local_iteration}" -gt "${iteration}" ]]; then
    #            iteration=${local_iteration}
    #        fi
    #    fi
    # done
    # A more robust way if the files are on a shared filesystem accessible by the submission node:
    if [ -f "$iteration_file" ]; then # Check on shared filesystem
         local_iteration_val=$(cat $iteration_file)
         if [[ "${local_iteration_val}" =~ ^[0-9]+$ ]]; then
            iteration=${local_iteration_val}
         fi
    fi
    echo "Found latest iteration: ${iteration}"

    if [[ $iteration -gt 0 ]]; then
        iteration_2="global_step${iteration}"
        # ds_ssh "echo $iteration > $iteration_file"
        # ds_ssh "echo $iteration_2 > $iteration_file_2"
        # If on shared filesystem, rank 0 (or this script) can write it once.
        echo $iteration > $iteration_file
        echo $iteration_2 > $iteration_file_2
        echo "Updated latest checkpoint files to iteration ${iteration}."
    fi
fi


echo "START!"
echo "--- PBS_NODEFILE ($PBS_NODEFILE) content ---"
cat $PBS_NODEFILE # Useful for debugging node allocation
echo "--- Unique nodes in $UNIQUE_NODES_FILE ---"
cat $UNIQUE_NODES_FILE
echo "--- nvidia-smi (on submission node) ---"
nvidia-smi

# MASTER_ADDR will be the first unique hostname from PBS_NODEFILE
if [ -s $UNIQUE_NODES_FILE ]; then
    export MASTER_ADDR=$(head -n 1 $UNIQUE_NODES_FILE)
else
    export MASTER_ADDR=$(hostname) # Fallback for local/single node
fi
export MASTER_PORT=29500 # Choose a free port
export WANDB_DEBUG=true # Propagate this

# The script `pretrain_bert.py` will be launched by `torchrun` on each node.
# `mpirun` launches `torchrun` once per node.
# `torchrun` launches `num_gpus_pernode` Python processes on that node.

# Construct the command to be executed by mpirun on each node
PYTHON_SCRIPT_PATH="/work/gg17/a97006/250519_modern_bert_0/Inhouse-Megatron-DeepSpeed/pretrain_modern_bert.py"
LOG_FILE="${log_path}/${jobname}_${host}_${current_time}.log" # mpirun will pipe output here from rank 0 of its direct children

echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "Number of Nodes for torchrun: ${num_node}"
echo "Processes per Node for torchrun: ${num_gpus_pernode}"
echo "Python script: ${PYTHON_SCRIPT_PATH}"
echo "Logging to: ${LOG_FILE}"

HOST=`head -n 1 ${PBS_NODEFILE}`
HOST_IP=$(getent hosts $HOST | awk '{print $1}')
export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.6"
MPIRUN_OPTIONS="--hostfile ${UNIQUE_NODES_FILE} -np ${num_node} -npernode 1 --map-by node -x CUDA_HOME -x LD_LIBRARY_PATH -x PATH"
# MPIRUN_OPTIONS はお使いのMPI実装に合わせてカスタマイズ可能です。例: OpenMPIの場合
# MPIRUN_OPTIONS="--hostfile ${UNIQUE_NODES_FILE} -np ${num_node} --map-by node -report-bindings"

# mpirunによって起動される各ノード上の ${num_node} 個のプロセスが、以下のtorchrunコマンドを実行します:
TORCHRUN_CMD="torchrun \
    --nnodes ${num_node} \
    --nproc_per_node ${num_gpus_pernode} \
    --rdzv_id ${PBS_JOBID:-$RANDOM} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${HOST_IP}:${MASTER_PORT} \
    ${PYTHON_SCRIPT_PATH} \
    ${megatron_options} \
    ${data_options} \
    ${deepspeed_options}"

echo "--------------------------------------------------------------------------"
echo "mpirun ${MPIRUN_OPTIONS} \\"
echo "${TORCHRUN_CMD}"
echo "--------------------------------------------------------------------------"
echo $CUDA_HOME
echo "--------------------------------------------------------------------------"
echo "HOST: $HOST"
echo "HOST_IP: $HOST_IP"
echo "--------------------------------------------------------------------------"
echo "mpirun ${OMPI_MCA_mca_base_env_list} \\"

export CUDA_HOME="/work/opt/local/aarch64/cores/cuda/12.6"
unset OMPI_MCA_mca_base_env_list
mpirun ${MPIRUN_OPTIONS} \
    ${TORCHRUN_CMD} &>> "${LOG_FILE}"

# Example if you want separate logs per rank of mpirun (which are torchrun instances)
# mpirun ${MPIRUN_OPTIONS} \
#    -x ... \
#    sh -c "${TORCHRUN_CMD} &>> \"${log_path}/${jobname}_rank_\${OMPI_COMM_WORLD_RANK}_${current_time}.log\""

rm -f $UNIQUE_NODES_FILE # Clean up temporary file for unique nodes
echo "END OF SCRIPT"