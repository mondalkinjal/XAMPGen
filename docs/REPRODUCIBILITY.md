# Reproduce the cached analyses

From the repository root, with the recorded scientific Python environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/reproduce_cached.py --output reproduced
```

The runner checks all 36 layers of attention-derived helicity scores, the selected
layer and its correlation, and the five-seed probe summary. It also produces the
probe figures and both 18-peptide novelty tables from saved alignments. Outputs
are written under `reproduced/`.

The recorded check is in [cached_verification.json](cached_verification.json).
Token mappings, small MIC scalers, saved probe results, and attention caches are
included. The original, single-file XAMPGen checkpoint and its download status
are described in [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md). Obtain pretrained
ProGen from the [original authors](https://doi.org/10.1038/s41587-022-01618-2). The Swiss-Prot database
is excluded. Checkpoint
training, peptide generation, full candidate screening, and new BLAST searches
have not been independently rerun for this release.

Run individual research scripts from `sequence_gen/`. Several figure scripts
retain the original manuscript output path, and SLURM launchers retain cluster
paths and environment names. The cached runner creates its own output layout.
The recorded environment is not a complete installation specification for every
training or external structure-prediction tool. A pinned environment for the
released ProGen training/generation scripts is now documented in
[TRAINING_GENERATION.md](TRAINING_GENERATION.md), with a small CPU LoRA training
check and full checkpoint tensor-name/shape checks.

Generation and interpretation use the original `finetuned_progen.pth`.
The separate local `finetuned_progen_full.pth` is not distributed by this release.
The correspondence between a historical training job and a specific local
checkpoint has not been independently established.

## Unknown-peptide attention extraction

The original cache producer and its sequence-list dependency are now included.
See [UNKNOWN_PEPTIDE_ATTENTION.md](UNKNOWN_PEPTIDE_ATTENTION.md) for extracting
all-layer attention from the original fine-tuned checkpoint, the exact plotting
transformation, and rendering the P1-P18 figures from a new or supplied cache.
