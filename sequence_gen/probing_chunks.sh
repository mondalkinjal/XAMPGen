#!/bin/bash
#SBATCH --job-name=probe-chunks
#SBATCH --account=energybio-eng
#SBATCH --partition=standard
#SBATCH -t 12:00:00
#SBATCH -n 1 -c 8
#SBATCH --mem=24G
#SBATCH -o slurm-probe-%x-%A_%a.out
# array index = 12*seed + block; block b covers layers 3b..3b+2.  TASK=cls|reg from --export.
source "/scratch/zt1/project/energybio/user/kinjal14/miniforge3/etc/profile.d/conda.sh"
conda activate sequence_generation_new
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
cd /scratch/zt1/project/energybio/user/kinjal14/AMP_Discovery/sequence_gen
SEED=$(( SLURM_ARRAY_TASK_ID / 12 )); B=$(( SLURM_ARRAY_TASK_ID % 12 ))
python3 probing_multiseed.py --task "$TASK" --seed "$SEED" --layers "$((3*B))-$((3*B+2))"
