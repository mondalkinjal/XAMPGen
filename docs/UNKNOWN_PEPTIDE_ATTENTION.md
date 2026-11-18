# Attention extraction for the unknown/generated peptide panel

The P1-P18 maps start with a forward pass of each complete peptide through the
original `finetuned_progen.pth`. The source that creates the cache is
[`top10_layers.py::scan()`](../sequence_gen/top10_layers.py), using the P1-P18
sequences in [`helicity_scan.py::PEPTIDES`](../sequence_gen/helicity_scan.py).
That scan also includes eight reference peptides. Both original source files
are included unchanged.

The measured helicity values stored alongside the peptide sequences are ignored
by the extraction loop. They enter the later helicity-correlation analysis.

## From a sequence to attention probabilities

1. `motif_head_lib.load_model()` builds the 36-layer, 16-head ProGen/CTRL model,
   attaches rank-16 LoRA adapters to `Wq` and `Wk`, loads `checkpoint['model']`
   from `finetuned_progen.pth` with strict matching, and selects evaluation mode.
   It uses CUDA when available and otherwise CPU.
2. `motif_head_lib.encode(sequence)` reads the released mappings and constructs
   `[Antimicrobial control token] + [amino-acid tokens]`. Token 0 is the control
   token; sequence residue 1 is token 1. This path adds no stop token or padding.
3. `motif_head_lib.run()` runs the entire encoded peptide with `past=None` and
   gradients disabled. It returns the attention from every layer through
   `fast_transformer_extraction.py`. This is a forward pass of an already chosen
   sequence, separate from autoregressive candidate generation.

For layer `l` and head `h`, the implementation computes

```text
Q = Linear_Wq(X),  K = Linear_Wk(X)
A[l,h] = softmax_keys(Q K^T / sqrt(d_k) + causal_mask)
d_k = 1280 / 16 = 80
```

The linear projections include their biases and attached LoRA updates. The
causal mask prevents query token
`i` from attending to a later key token `j > i`. Rows are queries and columns are
keys. Raw probabilities sum to one over the allowed keys in each row.

Each `all_attn[l]` has shape `[1, 16, N, N]` for one peptide, where
`N = sequence length + 1`. The original scan averages the 16 heads:

```python
out[name] = np.stack([
    a[0].float().mean(0).cpu().numpy() for a in all_attn
])
```

Thus each cache entry is `[36, N, N]`. The cache contains raw, head-averaged
attention probabilities for all layers. It does not preserve individual heads;
use `all_attn[l][0, h]` before averaging when a head-specific matrix is needed.

## Recompute the cache from the original checkpoint

Create the [training/generation environment](TRAINING_GENERATION.md), then add
the pinned analysis dependencies from the repository root:

```bash
conda activate xampgen-training-generation
python -m pip install -r requirements-cached.txt
cd sequence_gen
```

The original checkpoint must be present as `sequence_gen/finetuned_progen.pth`,
and `mapping_files/` must contain the released mappings. See
[model-weight availability](MODEL_WEIGHTS.md). The pretrained ProGen checkpoint
is not separately needed for this extraction.

This calls the original extraction function and writes a new cache, preserving
the included cache:

```bash
python - <<'PY'
from pathlib import Path
import top10_layers as extraction

cache = Path('top10_layers_recomputed/attn_all_layers.npz')
cache.parent.mkdir(parents=True, exist_ok=True)
extraction.CACHE = str(cache)
extraction.scan()
PY
```

The output has 26 entries: eight references and the 18 measured-panel peptides.
Keys retain their original labels, including `P9 (WT)`. The figure script maps
that key to its displayed label `P9` by taking the first word.

The original full command is `python top10_layers.py --rescan --no-figs`. It
replaces `top10_layers/attn_all_layers.npz` and computes additional layer-comparison
claim scores in `top10_layers/claims.csv`. The extraction-only command above
does not run those additional analyses. The preserved `top10_layers.sh` is an
optional site-specific Slurm launcher; adapt its account, environment, and paths
before using it on another system.

## How the heatmap values are obtained

`ordered_set_figs.py` selects cache index `22`, then applies
`motif_common.sym()`. Layer indices in these scripts are zero-based: cache index
22 is the 23rd transformer block.

For the head-averaged raw attention `A`, the plotting transformation is

```text
U[i,j] = 1/(i+1) when j <= i, otherwise 0
S[i,j] = (A[i,j] - U[i,j]) when j <= i, otherwise 0
C      = S + S^T, with the diagonal divided by 2
```

The baseline is uniform attention over the `i+1` causally available keys. The
transformation subtracts that baseline and reflects the causal triangle. The
displayed `C` is a signed deviation from uniform causal attention; it is not the
raw probability matrix. It can contain negative values and its rows need not
sum to one. The raw probabilities remain unchanged in the cache.

The residue map displays `C[1:, 1:]`. The separate AMP-tag strip displays
`C[0, 1:]`, with its own color scale. Because the raw attention is causal, this
strip reflects residues attending to the control token after baseline subtraction.

To render the three P1-P18 panels from the recomputed cache into a new folder:

```bash
python - <<'PY'
from pathlib import Path
import ordered_set_figs as figures

figures.CACHE = 'top10_layers_recomputed/attn_all_layers.npz'
figures.OUT = 'unknown_attention_figures'
Path(figures.OUT).mkdir(parents=True, exist_ok=True)
figures.main()
PY
```

This produces `ordered_set1_split.png`, `ordered_set2_split.png`, and
`ordered_set3_split.png`. To use the included cache instead, set `figures.CACHE`
to `top10_layers/attn_all_layers.npz`; rendering then requires no checkpoint.

## Extract values for an additional peptide

Run from `sequence_gen/`, with the original checkpoint and mappings in place:

```python
import numpy as np
import torch
import motif_head_lib as L

sequence = 'KWRSFIRGVTHAAHTALKHYFRRKRSG'  # Replace with the peptide to inspect.
unsupported = set(sequence.upper()) - set(L.AA_TO_IDX)
if unsupported:
    raise ValueError(f'Unsupported residues: {sorted(unsupported)}')

model, device = L.load_model()
tokens = torch.tensor([L.encode(sequence)], device=device)
_, all_attn = L.run(model, tokens)

per_head = np.stack([a[0].float().cpu().numpy() for a in all_attn])
head_mean = np.stack([a[0].float().mean(0).cpu().numpy() for a in all_attn])
np.savez('my_peptide_attention.npz', per_head=per_head, head_mean=head_mean)
```

This saves `[36, 16, N, N]` per-head probabilities and `[36, N, N]` head means.
The explicit residue check matters because the original `encode()` skips
characters absent from the amino-acid mapping.

## Verification scope

The included cache was checked for the 26 expected keys, shapes consistent with
36 layers and the extra control token, finite nonnegative probabilities,
causal masking, and probability row sums. See
[unknown-attention-validation.json](unknown-attention-validation.json).
The original checkpoint forward pass was not rerun for this documentation update;
the matching keys and shapes do not establish the historical checkpoint identity
of an existing cache. The scientific source files and the supplied cache remain
unchanged.
