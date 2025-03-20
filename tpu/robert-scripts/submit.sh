#!/bin/bash

#SBATCH --job-name=llm-finetuning
#SBATCH --time=1-00:00:00
#SBATCH --partition=a3
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=1800G
#SBATCH --gpus-per-task=8

NODES=( $( scontrol show hostnames ${SLURM_JOB_NODELIST} ) )
NODES_ARRAY=(${NODES})
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IPS=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)
HEAD_NODE_IPS_ARRAY=(${HEAD_NODE_IPS})
HEAD_NODE_IP=${HEAD_NODE_IPS_ARRAY[0]}

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "NODES=${NODES}"
echo "NODES_ARRAY=${NODES_ARRAY}"
echo "HEAD_NODE=${HEAD_NODE}"
echo "HEAD_NODE_IPS=${HEAD_NODE_IPS}"
echo "HEAD_NODE_IPS_ARRAY=${HEAD_NODE_IPS_ARRAY}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"

RUN_ID="$(date +%s)"

SCRATCH_PREFIX="/scratch-weka/${USER}"

OUTPUT_PATH="${SCRATCH_PREFIX}/finetuning/output/llama-3-1/${RUN_ID}"
mkdir -p "${OUTPUT_PATH}"

export HF_HOME=${SCRATCH_PREFIX}/.cache/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCH_LOGS=all
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_NCCL_TRACE_BUFFER_SIZE=32
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_TRACE_CPP_STACK=1
# export NCCL_DEBUG=INFO

srun --label accelerate launch \
	--config_file=fsdp_config.yaml \
	--mixed_precision=bf16 \
	--num_machines="${SLURM_NNODES}" \
	--num_processes="$((8 * SLURM_NNODES))" \
	--rdzv_backend=c10d \
	--main_process_ip="${HEAD_NODE_IP}" \
	--main_process_port=29500 \
	"${HOME}/run.py" \
	--output_dir="${OUTPUT_PATH}" \
	--lr_scheduler_type=constant_with_warmup \
	--learning_rate=0.000005 \
	--weight_decay=0.01 \
	--warmup_ratio=0.05 \
	--num_train_epochs=1 \
	--max_steps=-1 \
	--save_strategy=no \
	--dataloader_num_workers=1 \
	--per_device_train_batch_size=1 \
	--log_level=info \
	--logging_steps=0.01 \
	--disable_tqdm=True