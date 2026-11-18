# Included scripts

The release contains 35 Python files and 11 Bash files. Supporting modules
are included to preserve local imports. Research-script contents are preserved
from the reviewed local sources; release utilities are listed separately below.

- [sequence_gen/ProteinDataset.py](../sequence_gen/ProteinDataset.py)
- [sequence_gen/correlation_map.py](../sequence_gen/correlation_map.py)
- [sequence_gen/esmfold_bench.py](../sequence_gen/esmfold_bench.py)
- [sequence_gen/fast_transformer.py](../sequence_gen/fast_transformer.py)
- [sequence_gen/fast_transformer_extraction.py](../sequence_gen/fast_transformer_extraction.py)
- [sequence_gen/fast_transformer_linear_probing.py](../sequence_gen/fast_transformer_linear_probing.py)
- [sequence_gen/finetune_progen_lora_multigpu.py](../sequence_gen/finetune_progen_lora_multigpu.py)
- [sequence_gen/finetune_progen_lora_multigpu_h100.py](../sequence_gen/finetune_progen_lora_multigpu_h100.py)
- [sequence_gen/fold_helix_dssp.py](../sequence_gen/fold_helix_dssp.py)
- [sequence_gen/generate_amp_finetuned.py](../sequence_gen/generate_amp_finetuned.py)
- [sequence_gen/generate_amp_pretrained.py](../sequence_gen/generate_amp_pretrained.py)
- [sequence_gen/generation_analysis.py](../sequence_gen/generation_analysis.py)
- [sequence_gen/helicity_layer_profile.py](../sequence_gen/helicity_layer_profile.py)
- [sequence_gen/known_L22_figs.py](../sequence_gen/known_L22_figs.py)
- [sequence_gen/linear_probing_overall.py](../sequence_gen/linear_probing_overall.py)
- [sequence_gen/motif_common.py](../sequence_gen/motif_common.py)
- [sequence_gen/motif_fig3.py](../sequence_gen/motif_fig3.py)
- [sequence_gen/motif_figs.py](../sequence_gen/motif_figs.py)
- [sequence_gen/motif_head_lib.py](../sequence_gen/motif_head_lib.py)
- [sequence_gen/motif_scan.py](../sequence_gen/motif_scan.py)
- [sequence_gen/novelty_analysis/align_tables.py](../sequence_gen/novelty_analysis/align_tables.py)
- [sequence_gen/ordered_set_figs.py](../sequence_gen/ordered_set_figs.py)
- [sequence_gen/polyK_difference.py](../sequence_gen/polyK_difference.py)
- [sequence_gen/probing_4panel.py](../sequence_gen/probing_4panel.py)
- [sequence_gen/probing_multiseed.py](../sequence_gen/probing_multiseed.py)
- [sequence_gen/probing_multiseed_summary.py](../sequence_gen/probing_multiseed_summary.py)
- [sequence_gen/pytorch_transformer_gpu.py](../sequence_gen/pytorch_transformer_gpu.py)
- [sequence_gen/split_amp_figs.py](../sequence_gen/split_amp_figs.py)
- [sequence_gen/transformProtein.py](../sequence_gen/transformProtein.py)
- [sequence_gen/triad_split_figs.py](../sequence_gen/triad_split_figs.py)
- [tools/reproduce_cached.py](../tools/reproduce_cached.py)
- [sequence_gen/esmfold_bench.sh](../sequence_gen/esmfold_bench.sh)
- [sequence_gen/known_L22_figs.sh](../sequence_gen/known_L22_figs.sh)
- [sequence_gen/novelty_analysis/blast_align.sh](../sequence_gen/novelty_analysis/blast_align.sh)
- [sequence_gen/novelty_analysis/verify.sh](../sequence_gen/novelty_analysis/verify.sh)
- [sequence_gen/probing_chunks.sh](../sequence_gen/probing_chunks.sh)
- [sequence_gen/probing_multiseed.sh](../sequence_gen/probing_multiseed.sh)
- [sequence_gen/probing_reg_cache.sh](../sequence_gen/probing_reg_cache.sh)
- [sequence_gen/reproduce_figures.sh](../sequence_gen/reproduce_figures.sh)
- [sequence_gen/submit_progen_lora_a100_4gpu.sh](../sequence_gen/submit_progen_lora_a100_4gpu.sh)
- [sequence_gen/submit_progen_lora_h100_4gpu.sh](../sequence_gen/submit_progen_lora_h100_4gpu.sh)

## Attention extraction and release utilities

- [sequence_gen/top10_layers.py](../sequence_gen/top10_layers.py): original all-layer attention-cache producer.
- [sequence_gen/helicity_scan.py](../sequence_gen/helicity_scan.py): original P1-P18 sequences and layer-22 companion scan.
- [sequence_gen/top10_layers.sh](../sequence_gen/top10_layers.sh): original Slurm launcher.
- [tools/check_training_environment.py](../tools/check_training_environment.py): environment and checkpoint checks.
- [tools/download_model_weights.py](../tools/download_model_weights.py): download/verify the original single-file checkpoint.

Extraction and plotting steps are explained in [UNKNOWN_PEPTIDE_ATTENTION.md](UNKNOWN_PEPTIDE_ATTENTION.md).
