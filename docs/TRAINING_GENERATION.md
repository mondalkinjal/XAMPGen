# Training and generation environment

From the repository root, create the Linux/NVIDIA environment:

```bash
conda env create -f environment-training-generation.yml
conda activate xampgen-training-generation
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/check_training_environment.py
```

The environment pins Python 3.11.14, PyTorch 2.10.0 with CUDA 12.8, PEFT 0.18.1,
Transformers 5.2.0, Accelerate 1.12.0, NumPy 2.4.2, and TensorBoard 2.20.0.
These are pinned direct dependencies, not a lock of every transitive dependency.
The environment is for the released ProGen training and generation scripts;
external structure predictors and the separate classifier/MIC pipelines require
their own installations. For cached figure analysis use `requirements-cached.txt`.

The CUDA wheel includes its CUDA runtime; an NVIDIA driver compatible with CUDA
12.8 must already be installed on the execution host. The generation implementation
requires CUDA. On a GPU allocation, check it before running:

```bash
python tools/check_training_environment.py --require-cuda
```

The environment checker runs one small CPU LoRA/Accelerate optimization step and
writes a TensorBoard event. It does not train the full ProGen model. Its optional
checkpoint check verifies every tensor name and shape against the released
architecture without allocating the complete model:

```bash
python tools/check_training_environment.py \
  --checkpoint-dir "$PWD/sequence_gen" \
  --output training-generation-validation.json
```

Recorded validation is in [training-generation-validation.json](training-generation-validation.json).
This is a reconstructed environment, not a claim about the exact historical
environment or scheduler job used to produce each checkpoint. Full CUDA generation
and multi-GPU fine-tuning have not been rerun for this environment release.

## Model checkpoints

See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for the checkpoint inventory, checksums,
publication status, and restore command. Files must retain their documented names
under `sequence_gen/`. The two fine-tuned files are distinct and must not be
substituted for each other based on filename alone.

## Fine-tuning

Run from `sequence_gen/` so vocabulary, mappings, and sequences resolve correctly.
The released A100 variant uses the following parameters by default; explicit
output paths below keep new training runs separate from the released weights:

```bash
cd sequence_gen
accelerate launch --multi_gpu --num_processes 4 finetune_progen_lora_multigpu.py \
  --model_dir pretrain_progen_full.pth \
  --finetuned_model_dir finetuned_progen_retrained.pth \
  --num_epochs 30 --sequence_len 150 --batch_size 64
```

The H100 variant can be run with:

```bash
accelerate launch --multi_gpu --num_processes 4 finetune_progen_lora_multigpu_h100.py \
  --model_dir pretrain_progen_full.pth \
  --finetuned_model_dir finetuned_progen_h100_retrained.pth \
  --num_epochs 100 --sequence_len 150 --batch_size 64
```

The batch size is per process. Allocate sufficient GPU and host memory for the
full model and this batch size, or lower it for your hardware. The original Slurm
launchers retain site-specific paths, environment names, and resource requests;
adapt those settings before submission. The launchers' unused `SCRIPT_ARGS`
variable does not pass mixed-precision options to Accelerate. The commands above
leave precision at its default. See [MULTIGPU_LORA.md](MULTIGPU_LORA.md).

## Candidate generation

With `finetuned_progen.pth` restored, run a small example on a CUDA GPU from
`sequence_gen/`:

```bash
python generate_amp_finetuned.py --code 3 --num_sample_batches 1 \
  --batch_size 1 --gen_length 50 --top_p 0.5 --top_k 0 \
  --rep_penalty 1.2 --fname generated_amp
```

The output is `generated_amp_code_3.txt`. For the pretrained comparison:

```bash
python generate_amp_pretrained.py --code 3 --num_sample_batches 1 \
  --batch_size 1 --gen_length 50 --top_p 0.5 --top_k 0 \
  --rep_penalty 1.2 --fname generated_pretrained
```

Generation restores the fine-tuned checkpoint's optimizer state even though it
does not update the model. Preserve the full checkpoint for this original entry
point. The `--code` argument names the output; the saved scripts obtain the
Antimicrobial conditioning tag from the mappings. See the [generation README](../sequence_gen/README.md)
for sampling details.
