#!/bin/bash
#SBATCH --job-name=probe-reg-cache
#SBATCH --account=energybio-eng
#SBATCH --partition=standard
#SBATCH -t 06:00:00
#SBATCH -n 1 -c 64
#SBATCH --mem=160G
#SBATCH -o slurm-probe-reg-cache-%A_%a.out
# seed = array index; run index 0 first so it caches the embeddings for the others
source "/scratch/zt1/project/energybio/user/kinjal14/miniforge3/etc/profile.d/conda.sh"
conda activate sequence_generation_new
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
cd /scratch/zt1/project/energybio/user/kinjal14/AMP_Discovery/sequence_gen
python3 probing_multiseed.py --task reg --seed 0 --layers 1-0
