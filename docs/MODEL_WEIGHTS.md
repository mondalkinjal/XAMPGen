# Model weights

**Publication status:** all three original checkpoints have been located, hashed,
and packaged locally. Binary release upload is pending an authenticated GitHub
CLI session. The public repository currently contains the inventory, checksums,
and restore tool; the binary weights are not yet publicly downloadable.

| Checkpoint | Bytes | Use |
| --- | ---: | --- |
| `pretrain_progen_full.pth` | 4,629,670,777 | Initialization for training and the pretrained generation comparison |
| `finetuned_progen.pth` | 4,665,433,262 | The checkpoint read by `generate_amp_finetuned.py` |
| `finetuned_progen_full.pth` | 4,665,439,062 | A distinct fine-tuned checkpoint; also the default output filename of the A100 training variant |

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

After obtaining the release parts, restore the generation checkpoint:

```bash
python tools/restore_model_weights.py --parts-dir /path/to/downloaded-parts \
  --model finetuned_progen.pth
```

Omit `--model` to restore all three. The default output directory is
`sequence_gen/`. Use `--output-dir` to choose another directory. Each part and the
assembled file are verified against the manifest. An existing matching checkpoint
is reused; an existing mismatched checkpoint is rejected rather than replaced.
Once public hosting is configured, `--base-url` can download missing parts from
a release asset directory. The manifest deliberately has no download URL until
publication has been verified.

Allow about 14 GB for all parts and another 14 GB for the restored checkpoints.
For generation with the fine-tuned model alone, allow about 9.4 GB for its parts
and restored file. These disk sizes are not GPU memory estimates.

To check the environment and checkpoint architecture, use the commands in
[TRAINING_GENERATION.md](TRAINING_GENERATION.md). Package imports and shape checks
do not establish that a complete multi-GPU training run or CUDA generation was
reproduced. Attribution for the upstream model/code is in [THIRD_PARTY.md](THIRD_PARTY.md).
