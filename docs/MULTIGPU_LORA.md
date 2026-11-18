# Original multi-GPU LoRA fine-tuning code

The original multi-GPU implementations are included under the descriptive names
below. Python contents are byte-identical to the original files. Bash changes
only update Python filename targets; all SLURM settings are preserved.
See [FILENAME_MAP.md](FILENAME_MAP.md) for the original names.

| Implementation | Launcher | Requested GPUs |
| --- | --- | --- |
| [finetune_progen_lora_multigpu.py](../sequence_gen/finetune_progen_lora_multigpu.py) | [submit_progen_lora_a100_4gpu.sh](../sequence_gen/submit_progen_lora_a100_4gpu.sh) | 4 A100 GPUs |
| [finetune_progen_lora_multigpu_h100.py](../sequence_gen/finetune_progen_lora_multigpu_h100.py) | [submit_progen_lora_h100_4gpu.sh](../sequence_gen/submit_progen_lora_h100_4gpu.sh) | 4 H100 GPUs |

`submit_progen_lora_a100_4gpu.sh` requests `#SBATCH --gpus=a100:4`, sets
`GPUS_PER_NODE=4`, and executes:

```bash
accelerate launch --num_processes $GPUS_PER_NODE finetune_progen_lora_multigpu.py
```

`finetune_progen_lora_multigpu.py` has an active LoRA configuration with rank 16, alpha 32,
dropout 0.05, and target modules `Wq` and `Wk`. It constructs `Accelerator()`,
prepares the model/optimizer/data loader with `accelerator.prepare(...)`, uses
`accelerator.backward(loss)`, saves through `accelerator.save(...)`, and ends
with active `Trainer(...)` and `training.train(args.num_epochs)` calls.
Its default epoch count is 30, sequence length 150, and local batch size 64.
Default input checkpoint: `pretrain_progen_full.pth`; default output checkpoint:
`finetuned_progen_full.pth`. These weights are distributed separately from source.

The H100 launcher starts `finetune_progen_lora_multigpu_h100.py` with four Accelerate processes.
Its active LoRA and distributed training structure is retained unchanged.

These sources establish the saved multi-GPU implementation and requested GPU
configuration. This review has not rerun the multi-GPU training or independently
matched a historical scheduler job to a particular checkpoint.

