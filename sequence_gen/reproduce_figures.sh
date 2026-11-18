#!/bin/bash
#SBATCH --job-name=motif-figs
#SBATCH --account=energybio-eng
#SBATCH --partition=serial
#SBATCH --qos=normal
#SBATCH -N 1 -n 8 --mem=64G -t 02:00:00
#SBATCH --requeue --open-mode=append
#SBATCH -o slurm-motif-figs-%j.out
#
# Four figure sets in motif_head_results/pooled_panels/:
#   triad_cell   17 GKA/GKT/GRA/GRT occurrences, best (layer, head)   6/6/5
#   triad_layer  the same 17, best layer, 16 heads averaged           6/6/5
#   h17_cell      8 peptides with a literal H at 17, best (layer,head) 6/2
#   h17_layer     the same 8, best layer, heads averaged               6/2
#
# motif_scan.py is the only forward pass and is skipped when the cache exists.
#   sbatch reproduce_figures.sh            # figures only
#   sbatch reproduce_figures.sh --rescan   # redo the forward pass first
set -euo pipefail
source "/scratch/zt1/project/energybio/user/kinjal14/miniforge3/etc/profile.d/conda.sh"
conda activate sequence_generation_new
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
cd /scratch/zt1/project/energybio/user/kinjal14/AMP_Discovery/sequence_gen

if [ "${1:-}" = --rescan ] || [ ! -f motif_head_results/attn_cache.npz ]; then
  time python3 -u motif_scan.py
else
  echo "[skip] attn_cache.npz present -- pass --rescan to rebuild"
fi

time python3 -u motif_figs.py all
echo "[$(date)] done"
