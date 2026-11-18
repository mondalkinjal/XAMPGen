#!/bin/bash
#SBATCH --job-name=esmfold
#SBATCH --account=energybio-eng
#SBATCH --partition=serial
#SBATCH --qos=normal
#SBATCH -N 1 -n 8 --mem=96G -t 06:00:00
#SBATCH -o slurm-esmfold-%j.out
set -euo pipefail
source "/scratch/zt1/project/energybio/user/kinjal14/miniforge3/etc/profile.d/conda.sh"
conda activate sequence_generation_new
cd /scratch/zt1/project/energybio/user/kinjal14/AMP_Discovery/sequence_gen
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
export HF_HOME=/scratch/zt1/project/energybio/user/kinjal14/AMP_Discovery/sequence_gen/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
time python3 -u esmfold_bench.py
echo "[$(date)] done"
