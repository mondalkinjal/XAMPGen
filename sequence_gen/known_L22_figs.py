"""Layer-22 head-mean attention (uniform prior removed, reflected) for the known
peptides, as two 2x2 figures elongated in y so every residue label fits there.

  known_L22_main_2x2.png   TP4, Magainin-2, Plectasin, Indolicidin
  known_L22_supp_2x2.png   TP4 K-tail, TP4 no tail, QLAQ..., LL-37
Human neutrophil peptide 2 is left out.  The three TP4 panels share one colour
scale (as in attention_known_combined.py); the rest use their own.

The forward pass is cached in known_L22_cache.npz and skipped when present.
    python3 known_L22_figs.py [--rescan]
"""
import os
import string
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from motif_common import sym

LAYER = 22
OUT = 'probing_results_known_peptides'
CACHE = os.path.join(OUT, 'known_L22_cache.npz')

PEPTIDES = {
    'TP4':         "FIHHIIGGLFSAGKAIHRLIRRRRR",
    'TP4 K-tail':  "FIHHIIGGLFSAGKAIHRLIKKKKK",
    'TP4 no tail': "FIHHIIGGLFSAGKAIHRLI",
    'QLAQ':        "QLAQALAAALAALAQGWRRRRR",
    'Plectasin':   "GFGCNGPWDEDDMQCHNHCKSIKGYKGGYCAKGGFVCKCY",
    'Indolicidin': "ILPWKWPWWPWRR",
    'Magainin-2':  "GIGKFLHSAKKFGKAFVGEIMNS",
    'LL-37':       "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
}
FIGS = [('known_L22_main_2x2.png', ['TP4', 'Magainin-2', 'Plectasin', 'Indolicidin']),
        ('known_L22_supp_2x2.png', ['TP4 K-tail', 'TP4 no tail', 'QLAQ', 'LL-37'])]
SHARED = ('TP4', 'TP4 K-tail', 'TP4 no tail')
# panel titles; the PEPTIDES keys stay as the cache keys
LABEL = {'TP4 K-tail': 'TP4-polyK', 'TP4 no tail': 'TP4-noR5', 'QLAQ': 'NPR5'}


def scan():
    import torch
    import motif_head_lib as L
    model, device = L.load_model()
    out = {}
    for name, seq in PEPTIDES.items():
        _, all_attn = L.run(model, torch.tensor([L.encode(seq)], device=device))
        out[name] = all_attn[LAYER][0].float().cpu().numpy()      # [16, L, L]
        print(f'[scan] {name:12s} len={len(seq)}', flush=True)
    np.savez(CACHE, **out)


os.makedirs(OUT, exist_ok=True)
if '--rescan' in sys.argv or not os.path.exists(CACHE):
    scan()
cache = np.load(CACHE)
M = {name: sym(cache[name]).mean(0) for name in PEPTIDES}

shared = (min(M[n].min() for n in SHARED), max(M[n].max() for n in SHARED))

# as large as Plectasin's 41 rows allow: ~0.9 in of panel height per residue label
FONT = 64
plt.rcParams.update({'font.size': FONT, 'axes.titlesize': FONT, 'axes.titleweight': 'bold',
                     'xtick.labelsize': FONT, 'ytick.labelsize': FONT, 'axes.linewidth': 2.5})

for fname, names in FIGS:
    fig, axes = plt.subplots(2, 2, figsize=(60, 80))
    for idx, (ax, name) in enumerate(zip(axes.flat, names)):
        seq = PEPTIDES[name]
        vmin, vmax = shared if name in SHARED else (M[name].min(), M[name].max())
        im = ax.imshow(M[name], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax,
                     ticks=np.arange(np.floor(vmin * 20) / 20, np.ceil(vmax * 20) / 20 + 0.05, 0.05))

        tokens = ['AMP'] + [f'{aa}{i + 1}' for i, aa in enumerate(seq)]
        pos = np.arange(len(tokens))
        ax.set_yticks(pos)                        # every residue in y
        ax.set_yticklabels(tokens)
        ax.set_xticks(pos[::3])                   # every third in x
        ax.set_xticklabels(tokens[::3], rotation=90)
        ax.set_title(f'{string.ascii_uppercase[idx]}   {LABEL.get(name, name)}', pad=8)
        ax.grid(False)
        print(f'{fname}  {string.ascii_uppercase[idx]} = {name}')
    fig.subplots_adjust(left=0.06, right=0.94, top=0.985, bottom=0.035, hspace=0.095, wspace=0.23)
    fig.savefig(os.path.join(OUT, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
