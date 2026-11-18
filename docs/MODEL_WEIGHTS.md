# Model weights

XAMPGen uses the author's original **`finetuned_progen.pth`** checkpoint for both
generation and interpretation. It is one unchanged PyTorch file containing the
model and optimizer states. It is not converted, compressed, split, or stripped.

**Download status:** the original single file is preserved locally. A public
Google Drive link for this 4.7 GB file is pending. No public download URL is
configured yet. The earlier split-file release has been withdrawn.

| File | Exact bytes | Use |
| --- | ---: | --- |
| `finetuned_progen.pth` | 4,665,433,262 | Generation, probing, attention, and motif interpretation |

Original-file SHA-256:

```text
b98ddeb00335390744c4c653c3c1f5b72da42a8d05fde4bf5703330e14a0da70
```

The same metadata is in [model_manifest.json](../models/model_manifest.json) and
[SHA256SUMS](../models/SHA256SUMS). Verify an existing original checkpoint with:

```bash
python tools/download_model_weights.py --verify-only /path/to/finetuned_progen.pth
```

Once its public Drive link is configured, download the whole file with:

```bash
python -m pip install -r requirements-model-download.txt
python tools/download_model_weights.py
```

Public Drive links are handled by the pinned `gdown` dependency, including
Drive confirmation pages for large files. The original file is verified against
its SHA-256 after download.

The default destination is `sequence_gen/finetuned_progen.pth`. A matching file
already present is verified and reused. A mismatched existing file is rejected
without replacing it. The downloader preserves the original bytes and checks
the full-file SHA-256 before accepting a download.

## Which checkpoint the interpretation scripts use

| Task | Checkpoint selection |
| --- | --- |
| Peptide generation | `generate_amp_finetuned.py` loads `finetuned_progen.pth` |
| Motif and known-peptide attention rescans | `motif_head_lib.py` defaults to `finetuned_progen.pth` |
| Multi-seed probing | `probing_multiseed.py` defaults to `finetuned_progen.pth` |
| Main classification/regression probing | Pass `--checkpoint finetuned_progen.pth` |
| Figures from saved attention/probe caches | No checkpoint needed |

The separate local file `finetuned_progen_full.pth` is preserved locally and is
not distributed here. The released generation and interpretation defaults use
the checkpoint listed above. Historical training-job provenance has not been
independently established for the local files.

## Pretrained ProGen

For the original pretrained ProGen checkpoint, refer to **Madani et al. (2023),
Large language models generate functional protein sequences across diverse
families**, [Nature Biotechnology](https://doi.org/10.1038/s41587-022-01618-2) and the authors' resources
linked from that paper. This repository does not redistribute the pretrained
ProGen checkpoint.

Training and the pretrained generation comparison expect a compatible PyTorch
state dictionary at `sequence_gen/pretrain_progen_full.pth`. Follow the original
authors' instructions for obtaining the pretrained model; the released XAMPGen
scripts require the tensor names and shapes expected by this implementation.
The fine-tuned generation and interpretation workflows load the full XAMPGen
checkpoint directly and do not require a separate pretrained file.

See [TRAINING_GENERATION.md](TRAINING_GENERATION.md) for the environment and run
commands, and [THIRD_PARTY.md](THIRD_PARTY.md) for attribution.
