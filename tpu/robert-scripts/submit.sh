#!/bin/bash
# Copyright 2025 "Google LLC"
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --job-name=llm-finetuning
#SBATCH --time=1-00:00:00
#SBATCH --partition=a3
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=1800G
#SBATCH --gpus-per-task=8

NODES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
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

SCRATCH_PREFIX=${HOME}

OUTPUT_PATH="${SCRATCH_PREFIX}/finetuning/output/llama-3-1/${RUN_ID}"
mkdir -p "${OUTPUT_PATH}"

# export HF_HOME=${SCRATCH_PREFIX}/.cache/huggingface
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

python3 -m pdb accelerate launch \
	--config_file=fsdp_config.yaml \
	--mixed_precision=bf16 \
	--rdzv_backend=c10d \
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

srun python3 examples/pytorch/language-modeling/run_clm.py --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 --per_device_train_batch_size 128 --do_train --output_dir "$../output/test-clm" --overwrite_output_dir --config_name "../config.json" --cache_dir "/tmp/cache" --tokenizer_name meta-llama/Meta-Llama-3.1-70B --block_size "4096" --optim adafactor --save_strategy no --logging_strategy no --fsdp "full_shard" --fsdp_config "../fsdp_config.json" --torch_dtype bfloat16 --dataloader_drop_last yes --flash_attention --max_steps "5" --disable_tqdm=true
