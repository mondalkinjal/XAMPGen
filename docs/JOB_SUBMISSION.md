# Job submission and Bash helpers

These are the launchers associated with the README analyses. Original paths,
Conda environments, account names, and resource requests are preserved; adapt
them to your cluster before submission. Run the LoRA launchers from `sequence_gen/`.

| File | Role |
| --- | --- |
| [esmfold_bench.sh](../sequence_gen/esmfold_bench.sh) | SLURM job |
| [known_L22_figs.sh](../sequence_gen/known_L22_figs.sh) | SLURM job |
| [blast_align.sh](../sequence_gen/novelty_analysis/blast_align.sh) | BLAST search or table verification |
| [verify.sh](../sequence_gen/novelty_analysis/verify.sh) | BLAST search or table verification |
| [probing_chunks.sh](../sequence_gen/probing_chunks.sh) | SLURM job |
| [probing_multiseed.sh](../sequence_gen/probing_multiseed.sh) | SLURM job |
| [probing_reg_cache.sh](../sequence_gen/probing_reg_cache.sh) | SLURM job |
| [reproduce_figures.sh](../sequence_gen/reproduce_figures.sh) | SLURM job |
| [submit_progen_lora_a100_4gpu.sh](../sequence_gen/submit_progen_lora_a100_4gpu.sh) | SLURM job |
| [submit_progen_lora_h100_4gpu.sh](../sequence_gen/submit_progen_lora_h100_4gpu.sh) | SLURM job |

`novelty_analysis/verify.sh` requires `XAMPGEN_SI_TEX` to point to a local copy
of the manuscript supporting-information TeX file. That manuscript is not
included in this code repository.
