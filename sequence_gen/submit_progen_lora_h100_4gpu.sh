#!/bin/bash
#SBATCH --job-name=multigpu
#SBATCH -t 24:00:00
#SBATCH -n 32
#SBATCH --mem=500000 # number of MP tasks
#SBATCH --gpus=h100:4
#SBATCH --partition=gpu



. "/scratch/zt1/project/energybio/user/kinjal14/miniconda/etc/profile.d/conda.sh"
conda activate ML
export GPUS_PER_NODE=4
######################

#export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
#export SCRIPT="${ACCELERATE_DIR}/examples/complete_nlp_example.py"
current_directory=$(pwd)

export SCRIPT_ARGS=" \
    --mixed_precision fp16 \
    --output_dir ${current_directory}/output \
    --with_tracking \
    "

accelerate launch --num_processes $GPUS_PER_NODE finetune_progen_lora_multigpu_h100.py
