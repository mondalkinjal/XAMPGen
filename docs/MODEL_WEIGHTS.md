# Model weights

**Publication status:** all three original checkpoints are publicly available in
the [XAMPGen model release](https://github.com/mondalkinjal/XAMPGen/releases/tag/model-weights-20261118). GitHub asset sizes and
SHA-256 hashes were verified for all 12 parts before publication.

| Checkpoint | Bytes | Use |
| --- | ---: | --- |
| `pretrain_progen_full.pth` | 4,629,670,777 | Initialization for training and the pretrained generation comparison |
| `finetuned_progen.pth` | 4,665,433,262 | Primary checkpoint for generation and default probing/attention/motif interpretation |
| `finetuned_progen_full.pth` | 4,665,439,062 | Optional distinct checkpoint; also the default output filename of the A100 training variant |

## Which checkpoint to use

Use **`finetuned_progen.pth`** for both generation and interpretation with the
released scripts:

| Task | Checkpoint selection |
| --- | --- |
| Peptide generation | `generate_amp_finetuned.py` loads `finetuned_progen.pth` |
| Motif and known-peptide attention rescans | `motif_head_lib.py` defaults to `finetuned_progen.pth` |
| Multi-seed probing | `probing_multiseed.py` defaults to `finetuned_progen.pth` |
| Main classification/regression probing | Pass `--checkpoint finetuned_progen.pth` |
| Figures from included attention/probe caches | No checkpoint needed |
| Pretrained comparison and training initialization | `pretrain_progen_full.pth` |

`finetuned_progen_full.pth` is optional for these workflows. It is retained as
a distinct existing checkpoint, not designated as the interpretation model.
This table describes the file selection in the released code.

The fine-tuned files contain `model` and `optimizer` states. The pretrained file
is a plain model state dictionary. All three use the released 36-layer,
1280-dimensional architecture and a vocabulary of 129,407 tokens. The fine-tuned
models use rank-16 LoRA adapters on `Wq` and `Wk`. Historical training-job provenance
has not been independently established. A checkpoint filename does not establish
which historical job produced it.

Each file is split into four ordered parts, each at most 1,500,000,000 bytes,
to fit GitHub's [release asset limit](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
The split preserves every byte, including optimizer states. File and part
SHA-256 hashes are recorded in [model_manifest.json](../models/model_manifest.json).
[SHA256SUMS](../models/SHA256SUMS) lists the individual parts.

From the repository root, download and restore the generation checkpoint:

```bash
python tools/restore_model_weights.py --model finetuned_progen.pth
```

If you already downloaded the release parts, pass `--parts-dir /path/to/downloaded-parts`
to reuse them. The tool needs only Python's standard library.

Omit `--model` to restore all three. The default output directory is
`sequence_gen/`. Use `--output-dir` to choose another directory. Each part and the
assembled file are verified against the manifest. An existing matching checkpoint
is reused; an existing mismatched checkpoint is rejected rather than replaced.
The manifest records the verified public release download directory. Missing
parts download automatically; `--base-url` optionally selects another mirror.

Allow about 14 GB for all parts and another 14 GB for the restored checkpoints.
For generation with the fine-tuned model alone, allow about 9.4 GB for its parts
and restored file. These disk sizes are not GPU memory estimates.

To check the environment and checkpoint architecture, use the commands in
[TRAINING_GENERATION.md](TRAINING_GENERATION.md). Package imports and shape checks
do not establish that a complete multi-GPU training run or CUDA generation was
reproduced. Attribution for the upstream model/code is in [THIRD_PARTY.md](THIRD_PARTY.md).
