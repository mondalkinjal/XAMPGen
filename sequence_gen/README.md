# Sequence generation with XAMPGen

These scripts generate candidate peptide sequences one amino-acid token at a time
using ProGen. The two main scripts use the same sampling procedure with different
model weights, allowing sequences from the AMP-fine-tuned model to be compared
with those from the original pretrained model.

## What each script does

| Script | Role | Model weights read |
| --- | --- | --- |
| [generate_amp_finetuned.py](generate_amp_finetuned.py) | Generates candidate AMPs using ProGen fine-tuned with LoRA on AMP sequences. | `finetuned_progen.pth` |
| [generate_amp_pretrained.py](generate_amp_pretrained.py) | Generates comparison sequences using pretrained ProGen without the LoRA adaptation. | `pretrain_progen_full.pth` |

The fine-tuned script attaches LoRA adapters to the query and key projections
(`Wq` and `Wk`), with rank 16, alpha 32, and dropout 0.05, before loading the saved
model state. It also restores the optimizer state stored in that checkpoint,
although generation uses evaluation mode and does not update the model weights.
The pretrained script loads the original model state directly.

## How a sequence is generated

Both scripts follow these steps:

1. Read the vocabulary and token mappings, which translate amino acids and
   functional tags into the numerical token IDs understood by ProGen.
2. Load the appropriate checkpoint, move the model to a CUDA GPU, and put it in
   evaluation mode.
3. Find the token associated with the **Antimicrobial** functional tag and use it
   as the starting context for every sequence in the batch.
4. Predict scores for the next token from the sequence generated so far. The
   transformer reuses cached attention information from earlier positions.
5. Apply the saved repetition-penalty calculation and top-k/top-p sampling
   settings, then select the next token. With `top_k=1`, the code instead selects
   the highest-scoring token.
6. Repeat until the requested generation-length budget is reached.
7. Convert token IDs back into sequence characters and truncate each output at
   its first stop marker (`!`), if one was generated.
8. Write one peptide sequence per line to a text file.

The generation loop uses a fixed token budget; the stop marker is handled when
the output sequences are decoded. The initial functional tag occupies one token
position in that budget.

## Inputs and sampling settings

The inputs are the model checkpoint and files in `mapping_files/`. These include
the vocabulary, amino-acid token IDs, and keyword/taxonomy mappings. The mappings
must correspond to the checkpoint's vocabulary. The large checkpoint files are
kept separately from this source repository. Use the original single-file
`finetuned_progen.pth`; see [model-weight availability](../docs/MODEL_WEIGHTS.md).
Obtain the pretrained ProGen checkpoint from the [original paper](https://doi.org/10.1038/s41587-022-01618-2)
and the authors' resources.

| Argument | What it controls in the saved scripts |
| --- | --- |
| `--num_sample_batches` | Number of batches of sequences to generate; explicitly required. |
| `--batch_size` | Number of sequences generated together; default 32. |
| `--gen_length` | Token-generation budget, including the initial tag; explicitly required. |
| `--top_p` | Nucleus-sampling probability threshold; default 0.5. |
| `--top_k` | Restricts sampling to the highest-scoring k tokens when positive; default 0. |
| `--rep_penalty` | Value used by the saved repetition-penalty calculation; default 1.2. |
| `--fname` | Output-file prefix; default `samples`. |
| `--code` | Required numeric argument used in the output filename. The actual starting context in these scripts is explicitly the Antimicrobial tag found in the mappings. |

## Outputs

Each script writes:

```text
<fname>_code_<code>.txt
```

For example, an output prefix of `generated_amp` and code `3` produces
`generated_amp_code_3.txt`. The file contains plain peptide sequences, one per
line, without FASTA headers. No deduplication, AMP-classifier screening, MIC
prediction, or experimental validation is performed by these two scripts.

## Supporting code

| File or directory | What it provides |
| --- | --- |
| [fast_transformer.py](fast_transformer.py) | Transformer layers and attention caching used for autoregressive generation. |
| [transformProtein.py](transformProtein.py) | Protein/token transformation code and keyword mappings imported by the generation scripts. |
| `mapping_files/` | Vocabulary and amino-acid, keyword, and taxonomy token mappings. |

The scripts read relative paths from the working directory, so their expected
working directory is this `sequence_gen/` folder. Both main generation scripts
use CUDA. Their Python source contents have been preserved; this README explains
the saved implementation. Generation with model checkpoints has not been rerun
as part of preparing this review folder.
