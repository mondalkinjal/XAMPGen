#!/bin/bash
#SBATCH --job-name=known-L22
#SBATCH --account=energybio-eng
#SBATCH --partition=serial
#SBATCH --qos=normal
#SBATCH -N 1 -n 8 --mem=64G -t 01:00:00
#SBATCH -o slurm-known-L22-%j.out
set -euo pipefail
source "/scratch/zt1/project/energybio/user/kinjal14/miniforge3/etc/profile.d/conda.sh"
conda activate sequence_generation_new
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
cd /scratch/zt1/project/energybio/user/kinjal14/AMP_Discovery/sequence_gen
time python3 -u known_L22_figs.py "$@"
