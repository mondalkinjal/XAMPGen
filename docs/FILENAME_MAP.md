# Descriptive filenames

Python contents were preserved when these files were renamed. Bash changes
only updated the target Python filenames. SLURM settings were preserved.

| Original path | Release path |
| --- | --- |
| `sequence_gen/working_code.py` | [sequence_gen/finetune_progen_lora_multigpu.py](../sequence_gen/finetune_progen_lora_multigpu.py) |
| `sequence_gen/working_code.sh` | [sequence_gen/submit_progen_lora_a100_4gpu.sh](../sequence_gen/submit_progen_lora_a100_4gpu.sh) |
| `sequence_gen/pytorch_check_ending.py` | [sequence_gen/finetune_progen_lora_multigpu_h100.py](../sequence_gen/finetune_progen_lora_multigpu_h100.py) |
| `sequence_gen/pytorch_check_ending.sh` | [sequence_gen/submit_progen_lora_h100_4gpu.sh](../sequence_gen/submit_progen_lora_h100_4gpu.sh) |
| `sequence_gen/amp_gen_finetuned.py` | [sequence_gen/generate_amp_finetuned.py](../sequence_gen/generate_amp_finetuned.py) |
| `sequence_gen/amp_gen_pretrained.py` | [sequence_gen/generate_amp_pretrained.py](../sequence_gen/generate_amp_pretrained.py) |
