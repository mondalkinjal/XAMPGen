# XAMPGen: what the code does

This repository accompanies **XAMPGen: An Interpretable Super Intelligence Pipeline
For Novel Antimicrobial Peptide Discovery**. The code follows this workflow:

```text
AMP sequence data -> LoRA fine-tuning -> candidate generation -> AMP/MIC screening
                                                               |
                                        synthesized/measured peptide set
                                                               |
                      layer probes -> attention maps -> motif interpretation
```

The complete script list is in [SOURCE_INDEX.md](docs/SOURCE_INDEX.md), and renamed
files are mapped to their original names in [FILENAME_MAP.md](docs/FILENAME_MAP.md).

## Training environment and model weights

The pinned [training/generation environment](environment-training-generation.yml)
and [run instructions](docs/TRAINING_GENERATION.md) cover the released ProGen scripts.
Use the original, single-file **`finetuned_progen.pth`** for generation and
interpretation. [MODEL_WEIGHTS.md](docs/MODEL_WEIGHTS.md) records its exact size,
SHA-256, and verification instructions. Download the original checkpoint from
[Google Drive](https://drive.google.com/file/d/1Z6G65XrvSzs_9oZLMR_rWbf8HZ7wOp_W/view?usp=sharing). The public download was verified against the
original full-file SHA-256; the earlier split-file distribution is withdrawn.

```bash
conda env create -f environment-training-generation.yml
conda activate xampgen-training-generation
python -m pip install -r requirements-model-download.txt
python tools/download_model_weights.py
```

For the pretrained ProGen checkpoint, refer to the [original ProGen paper](https://doi.org/10.1038/s41587-022-01618-2)
and the authors' resources. The pretrained weights are not redistributed here.

## Methods and Materials

### Fine-tuning of ProGen

These scripts adapt pretrained ProGen to AMP sequences using LoRA.

| File | What it does |
| --- | --- |
| [finetune_progen_lora_multigpu.py](sequence_gen/finetune_progen_lora_multigpu.py) | Attaches LoRA adapters to `Wq` and `Wk`, trains with Accelerate, and saves a checkpoint. |
| [submit_progen_lora_a100_4gpu.sh](sequence_gen/submit_progen_lora_a100_4gpu.sh) | Requests four A100 GPUs through SLURM and launches the matching training script with four Accelerate processes. |
| [finetune_progen_lora_multigpu_h100.py](sequence_gen/finetune_progen_lora_multigpu_h100.py) | Provides the other active multi-GPU LoRA variant, with different training defaults. |
| [submit_progen_lora_h100_4gpu.sh](sequence_gen/submit_progen_lora_h100_4gpu.sh) | Requests four H100 GPUs and launches that training variant. |
| [pytorch_transformer_gpu.py](sequence_gen/pytorch_transformer_gpu.py) | Implements the transformer used by the saved training scripts. |
| [ProteinDataset.py](sequence_gen/ProteinDataset.py) and [transformProtein.py](sequence_gen/transformProtein.py) | Turn sequence data into padded token sequences and training labels using the saved vocabulary and mappings. |

**Inputs:** `unique_combined_sequences.txt`, the corresponding token mappings,
and the pretrained ProGen checkpoint. **Outputs:** a fine-tuned model checkpoint,
optimizer state, and training logs. The default output checkpoint names and epoch
counts differ between the two saved variants; see [MULTIGPU_LORA.md](docs/MULTIGPU_LORA.md).
The exact historical training job that produced a particular checkpoint has not
been independently established by this documentation.

### Generation and filtering

| File or directory | What it does |
| --- | --- |
| [generate_amp_finetuned.py](sequence_gen/generate_amp_finetuned.py) | Loads `finetuned_progen.pth` and samples peptide sequences conditioned on the Antimicrobial tag. Uses the saved top-k/top-p and repetition-penalty calculations. |
| [generate_amp_pretrained.py](sequence_gen/generate_amp_pretrained.py) | Uses the same sampling procedure with the original pretrained ProGen model, providing comparison sequences. |
| [fast_transformer.py](sequence_gen/fast_transformer.py) | Supplies transformer layers and cached attention states for autoregressive generation. |

**Generation inputs:** a model checkpoint and `mapping_files/`.
**Generation output:** one candidate sequence per line in
`<fname>_code_<code>.txt`. Sampling itself does not apply the classifier or MIC
regressor. [The generation README](sequence_gen/README.md) explains the sampling
steps and arguments in more detail.

Filtering prioritizes candidates predicted to be antimicrobial and active at low
concentrations. Refer to these papers for the screening methods:

**AMP classification:** Cordoves-Delgado, G., and García-Jacas, C. R. (2024).
*Predicting Antimicrobial Peptides Using ESMFold-Predicted Structures and
ESM-2-Based Amino Acid Features with Graph Deep Learning.* Journal of Chemical
Information and Modeling, 64(10), 4310–4321.
[DOI: 10.1021/acs.jcim.3c02061](https://doi.org/10.1021/acs.jcim.3c02061).

**MIC regression:** Padi, S., et al. (2026).
*Species- and Topic-aware Representation Learning for Antimicrobial Peptide
Discovery.* bioRxiv preprint.
[DOI: 10.64898/2026.05.28.728246](https://doi.org/10.64898/2026.05.28.728246).

Classification and MIC-regression source are outside this release; use the
references above for those methods.

### Data Sets

| Input | How it is used |
| --- | --- |
| [unique_combined_sequences.txt](sequence_gen/unique_combined_sequences.txt) | AMP sequence input for the saved ProGen fine-tuning code and sequence-comparison analysis. |
| [classification.csv](sequence_gen/classification.csv) | Labeled classification data used by the probing code. |
| [E_coli_DBAASP_with_independent_partition4.csv](sequence_gen/E_coli_DBAASP_with_independent_partition4.csv) | E. coli MIC data with saved dataset partitions for regression-related analysis. |
| `mapping_files/` | Maps amino acids, functional keywords, and taxonomy information to the model vocabulary. |
| `mic_scaler.pkl` and `mic_scaler_all_sources.pkl` | Saved MIC scaling objects used by the corresponding regression variants. |
| [helicity_18.fasta](sequence_gen/helicity_18.fasta) | Sequences of the 18 measured peptides used in downstream comparisons. |

Measured MIC, fluorescence, and circular-dichroism values are used in the analyses
below. The repository includes derived measurements, but no assay acquisition,
raw-spectrum fitting, or unified upstream database-curation pipeline.

## Results

### Outcomes of the fine-tuning process

[generation_analysis.py](sequence_gen/generation_analysis.py) compares sequences
from the training data, fine-tuned model, and pretrained model. It calculates
sequence length, isoelectric point, GRAVY hydrophobicity, helical/sheet
hydrophobic moments, and amino-acid composition.

**Inputs:** `unique_combined_sequences.txt`,
`amp_samples_finetuned_code_1.txt`, and `amp_samples_pretrained_code_1.txt`.
The generated sequence files must be supplied separately or produced by sampling.
**Outputs:** physicochemical-property violin plots and individual/grouped
amino-acid composition plots. The saved preprocessing replaces non-standard
residue symbols according to the mapping defined in that script.

## XSI analysis

### Layer probing

| File | What it does |
| --- | --- |
| [linear_probing_overall.py](sequence_gen/linear_probing_overall.py) | Extracts ProGen layer embeddings and trains/evaluates classification and MIC regression probes for individual layers. |
| [probing_multiseed.py](sequence_gen/probing_multiseed.py) | Repeats probing across seeds while caching the sequence embeddings, so repeats vary the probe initialization and batch ordering. |
| [probing_multiseed_summary.py](sequence_gen/probing_multiseed_summary.py) | Combines saved seed results into layer-wise summaries and plots. |
| [helicity_layer_profile.py](sequence_gen/helicity_layer_profile.py) | Computes a Lomb-Scargle periodicity score from attention-versus-residue-separation profiles, then correlates the scores with measured helicity at every layer. |
| [probing_4panel.py](sequence_gen/probing_4panel.py) | Combines the classification, regression, and helicity-probe results into the four-panel figure. |

**Inputs:** labeled sequences, MIC scaling inputs, the relevant model checkpoint,
or saved embeddings/attention/probe outputs, depending on the stage.
**Outputs:** layer-wise performance results, summaries, attention-derived helicity
scores, and figures. The cached helicity analysis recovers layer 22 with
Pearson r = 0.6946937160 for the measured peptide set.

For the structure-predictor comparison, [esmfold_bench.py](sequence_gen/esmfold_bench.py)
produces ESMFold structures for the peptide panel.
[fold_helix_dssp.py](sequence_gen/fold_helix_dssp.py) assigns secondary structure
from predicted backbone coordinates and compares the resulting helix fractions
with the measured helicities. Saved ESMFold/OmegaFold helicity arrays are included;
new structure predictions require the external predictor installations.

### Attention heatmap analysis on known antimicrobial peptides

[known_L22_figs.py](sequence_gen/known_L22_figs.py) extracts/caches layer-22 attention
for the known peptide panel and draws head-averaged maps.
[split_amp_figs.py](sequence_gen/split_amp_figs.py) renders these maps with the
Antimicrobial-tag attention shown as its own strip, separately from the
residue-residue matrix. These maps show residue-level patterns in TP4 and the
other reference peptides.

**Inputs:** reference peptide sequences and a fine-tuned checkpoint or the
corresponding saved attention cache. **Outputs:** labeled attention heatmaps.
[polyK_difference.py](sequence_gen/polyK_difference.py) additionally compares TP4
with its poly-lysine-tail variant through a difference map.

### Attention heatmap analysis on unknown peptides

The unknown/generated peptide attention values are extracted from the original
`finetuned_progen.pth` by [top10_layers.py](sequence_gen/top10_layers.py), using
the P1-P18 sequences in [helicity_scan.py](sequence_gen/helicity_scan.py).
The model reads the Antimicrobial control token followed by the peptide and
returns causal attention from all 36 layers and 16 heads. The extraction code
averages the heads and stores a `[36, length+1, length+1]` array per peptide in
`top10_layers/attn_all_layers.npz`.

[ordered_set_figs.py](sequence_gen/ordered_set_figs.py) selects cache index 22,
uses `motif_common.sym` to subtract the uniform causal baseline and reflect the
attention triangle, and draws the AMP-tag strip separately from the residue map.
These signed heatmap values differ from the raw attention probabilities in the
cache. Outputs are `ordered_set1_split.png`, `ordered_set2_split.png`, and
`ordered_set3_split.png`.

See [UNKNOWN_PEPTIDE_ATTENTION.md](docs/UNKNOWN_PEPTIDE_ATTENTION.md) for the
extraction commands, token indexing, equations, cache format, rendering steps,
and an example for an additional peptide. The helical-wheel interpretation uses
the externally produced MPEx diagrams cited in the paper.

### A more granular analysis at specific attention heads and layers

| File | What it does |
| --- | --- |
| [motif_head_lib.py](sequence_gen/motif_head_lib.py) | Loads the model and returns its individual layer/head attention tensors. |
| [motif_scan.py](sequence_gen/motif_scan.py) | Runs motif-bearing peptides through the model, saves their full attention stacks, and records the GKA/GKT/GRA/GRT occurrences. |
| [motif_common.py](sequence_gen/motif_common.py) | Defines the attention transformation and scoring functions used to inspect motif-specific signals. |
| [motif_fig3.py](sequence_gen/motif_fig3.py) | Selects and plots the layer/head with strongest triad attention for each motif-bearing peptide. |
| [triad_split_figs.py](sequence_gen/triad_split_figs.py) | Renders the selected triad maps with the Antimicrobial tag separated from residue-residue attention. |
| [motif_figs.py](sequence_gen/motif_figs.py) | Provides the saved triad/H17 head-specific and head-averaged figure variants. |

**Outputs:** attention caches and motif metadata under `motif_head_results/`,
plus head-specific attention figures. Once the required caches exist, the plotting
stages operate on saved attention without another model forward pass.

## Discussion and supplementary sequence-novelty analysis

The supplementary sequence-similarity comparisons use these scripts:

| File | What it does |
| --- | --- |
| [blast_align.sh](sequence_gen/novelty_analysis/blast_align.sh) | Runs BLASTP-short searches against the training AMP set and a local Swiss-Prot database, saving alignment details. |
| [align_tables.py](sequence_gen/novelty_analysis/align_tables.py) | Selects a returned hit per query and formats the aligned sequences, query-normalized identities, and E-values as LaTeX table rows. |
| [verify.sh](sequence_gen/novelty_analysis/verify.sh) | Checks the saved alignment-table identity calculations. |

**Inputs:** the measured peptide sequences, training AMP sequences, local
Swiss-Prot database for new searches, or the saved BLAST alignment TSV files.
**Outputs:** alignment records and the two supplementary sequence-comparison tables.

The table-rendering code ranks the returned BLAST hits by identical residues
**divided by the full query length**, using E-value to break ties. This differs
from selecting the lowest-E-value hit. The analysis describes similarity to the
searched references; it does not establish universal sequence novelty.

## Bash files for job submission

The job-submission and Bash helper files included in this release are listed in
[JOB_SUBMISSION.md](docs/JOB_SUBMISSION.md). Probe, attention, and structure
launchers retain their original cluster settings. Their presence does not by
itself establish which historical job generated a reported result.
