"""Re-run the layer-22 attention analysis at the ten best layers of the multi-seed probe.

For each layer in TOP10 (best mean overall rank first) this writes, under top10_layers/L<n>/,
  known_main_2x2.png, known_supp_2x2.png   as known_L22_figs.py (Fig. 4 and its SI figure)
  helicity_set{1,2,3}.png                  the 18 measured peptides, as helicity_figs.py
and, for all ten layers, top10_layers/claims.csv: every biological claim the manuscript reads
off the layer-22 maps, scored the same way at each layer.

Scores use the map M = sym(A) averaged over heads (the figures' map), token 0 = AMP tag:
  band(j) = mean_i M[i, j] over residues i != j      a residue's "broad band"
  pct     = percentile of a residue's band among the peptide's residues (100 = strongest);
            for several residues, their mean percentile; for a residue pair, the pair's
            percentile among all residue pairs.
    python3 top10_layers.py [--rescan]
"""
import os
import re
import string
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import lombscargle
from scipy.stats import pearsonr, spearmanr

from motif_common import sym, write_sets
from helicity_scan import PEPTIDES as MEASURED

# as in known_L22_figs.py (not imported: that module draws its figures on import)
KNOWN = {
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
LABEL = {'TP4 K-tail': 'TP4-polyK', 'TP4 no tail': 'TP4-noR5', 'QLAQ': 'NPR5'}

TOP10 = [23, 17, 33, 19, 30, 21, 28, 20, 27, 22]
OUT = 'top10_layers'
CACHE = os.path.join(OUT, 'attn_all_layers.npz')


def scan():
    import torch
    import motif_head_lib as L
    model, device = L.load_model()
    out = {}
    for name, seq in list(KNOWN.items()) + [(n, s) for n, s, _ in MEASURED]:
        _, attn = L.run(model, torch.tensor([L.encode(seq)], device=device))
        out[name] = np.stack([a[0].float().mean(0).cpu().numpy() for a in attn])  # [36, L, L], heads averaged
        print(f'[scan] {name:12s} len={len(seq)}', flush=True)
    np.savez(CACHE, **out)


# ---------- figures ----------
def known_figs(M, layer, outdir):
    """known_L22_figs.py's two 2x2 figures at another layer (dpi 100 for review)."""
    shared = (min(M[n].min() for n in SHARED), max(M[n].max() for n in SHARED))
    FONT = 64
    with plt.rc_context({'font.size': FONT, 'axes.titlesize': FONT, 'axes.titleweight': 'bold',
                         'xtick.labelsize': FONT, 'ytick.labelsize': FONT, 'axes.linewidth': 2.5}):
        for fname, names in FIGS:
            fig, axes = plt.subplots(2, 2, figsize=(60, 80))
            for idx, (ax, name) in enumerate(zip(axes.flat, names)):
                seq = KNOWN[name]
                vmin, vmax = shared if name in SHARED else (M[name].min(), M[name].max())
                im = ax.imshow(M[name], cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
                cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
                fig.colorbar(im, cax=cax)
                tokens = ['AMP'] + [f'{aa}{i + 1}' for i, aa in enumerate(seq)]
                pos = np.arange(len(tokens))
                ax.set_yticks(pos)
                ax.set_yticklabels(tokens)
                ax.set_xticks(pos[::3])
                ax.set_xticklabels(tokens[::3], rotation=90)
                ax.set_title(f'{string.ascii_uppercase[idx]}   {LABEL.get(name, name)}   (layer {layer})', pad=8)
            fig.subplots_adjust(left=0.06, right=0.94, top=0.985, bottom=0.035, hspace=0.095, wspace=0.23)
            fig.savefig(os.path.join(outdir, fname.replace('L22_', '')), dpi=100, bbox_inches='tight')
            plt.close(fig)


SET_GRID = dict(nrows=3, ncols=2, figsize=(6.5, 8.3), tick=8, ytick=8, xtick=8, cbtick=10, xstep=3,
                xrot=0, title_pad=1.5, cbar_pad=0.05, tick_pad=1.5, ystep=1, dpi=300,
                titles_in_gaps=True, rows_by_length=True,
                adjust=dict(left=0.06, right=0.94, top=0.985, bottom=0.015, hspace=0.06, wspace=0.32),
                rc={'font.size': 6, 'axes.titlesize': 8, 'axes.linewidth': 0.6,
                    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                    'xtick.major.size': 2, 'ytick.major.size': 2},
                pad_inches=0.02)


# ---------- claim scores ----------
def band(M):
    R = M[1:, 1:].copy()                      # residues only
    np.fill_diagonal(R, np.nan)
    return np.nanmean(R, axis=0)              # column means: band of residue j (1-based j -> index j-1)


def pct_res(M, residues):
    b = band(M)
    p = 100 * (b.argsort().argsort() + 0.5) / len(b)
    return float(np.mean([p[r - 1] for r in residues]))


def pct_pair(M, i, j):
    R = M[1:, 1:]
    iu = np.triu_indices(R.shape[0], 1)
    vals = R[iu]
    return float(100 * (vals < R[i - 1, j - 1]).mean())


def amp_pct(M, residues):
    a = M[1:, 0]
    p = 100 * (a.argsort().argsort() + 0.5) / len(a)
    return float(np.mean([p[r - 1] for r in residues]))


def offdiag_r(A, B):
    k = ~np.eye(A.shape[0], dtype=bool)
    return float(pearsonr(A[k], B[k])[0])


DMAX, PERIOD = 8, 3.6


def helicity_score(M):
    n, d = M.shape[0] - 1, np.arange(1, DMAX + 1, dtype=float)
    w = np.array([2 * np.pi / PERIOD])
    p = []
    for i in range(1, n + 1):
        prof = np.array([np.mean([M[i, j] for j in (i - k, i + k) if 1 <= j <= n])
                         for k in range(1, DMAX + 1)])
        p.append(np.atleast_1d(lombscargle(d, prof - prof.mean(), w, normalize=True))[0])
    return np.mean(p)


def motif_positions(seq, pattern):
    m = re.search(pattern, seq)
    return list(range(m.start() + 1, m.end() + 1)) if m else []


def claims(M, M22):
    tp4, pk, nt = M['TP4'], M['TP4 K-tail'], M['TP4 no tail']
    lvtg = [n for n in ('P2', 'P3', 'P4', 'P5', 'P6', 'P8', 'P11', 'P12')]
    pk_band = band(pk)
    c = {
        'TP4 polyR R21-25 band (pct)': pct_res(tp4, range(21, 26)),
        'TP4 polyR R21-25 AMP-tag (pct)': amp_pct(tp4, range(21, 26)),
        'TP4 H3 band (pct)': pct_res(tp4, [3]),
        'TP4 H17 band (pct)': pct_res(tp4, [17]),
        'TP4 F1 band (pct)': pct_res(tp4, [1]),
        'polyK: K22 strongest of K21-25': float(np.argmax(pk_band[20:25]) == 1),
        'polyK: K22 band (pct)': pct_res(pk, [22]),
        'polyK: R18-tail coupling, TP4 minus polyK': float(tp4[18, 21:26].mean() - pk[18, 21:26].mean()),
        'no-tail: core map r vs TP4 core': offdiag_r(tp4[:21, :21], nt),
        'NPR5 polyR R18-22 band (pct)': pct_res(M['QLAQ'], range(18, 23)),
        'NPR5 Q1,Q4 band (pct)': pct_res(M['QLAQ'], [1, 4]),
        'Magainin-2 G1,G3 band (pct)': pct_res(M['Magainin-2'], [1, 3]),
        'Plectasin F2,G3,C4,C37 band (pct)': pct_res(M['Plectasin'], [2, 3, 4, 37]),
        'Plectasin loop H16..K26 band (pct)': pct_res(M['Plectasin'], [16, 18, 20, 22, 23, 25, 26]),
        'Plectasin C4-C30 pair (pct)': pct_pair(M['Plectasin'], 4, 30),
        'Plectasin C19-C39 pair (pct)': pct_pair(M['Plectasin'], 19, 39),
        'Indolicidin W4,6,8,9,11 band (pct)': pct_res(M['Indolicidin'], [4, 6, 8, 9, 11]),
        'Indolicidin R12,R13 band (pct)': pct_res(M['Indolicidin'], [12, 13]),
        'LL-37 L1,L2 band (pct)': pct_res(M['LL-37'], [1, 2]),
        'LVTG/LITG motif band, P2-6,8,11,12 (pct)': np.mean(
            [pct_res(M[n], motif_positions(dict((a, s) for a, s, _ in MEASURED)[n], 'L[VI]TG')) for n in lvtg]),
        'W2 band, P2-P5 (pct)': np.mean([pct_res(M[n], [2]) for n in ('P2', 'P3', 'P4', 'P5')]),
    }
    x = np.array([helicity_score(M[n]) for n, _, _ in MEASURED])
    y = np.array([h for _, _, h in MEASURED])
    c['helicity Pearson r'] = float(pearsonr(x, y)[0])
    c['helicity Pearson p'] = float(pearsonr(x, y)[1])
    c['helicity Spearman rho'] = float(spearmanr(x, y)[0])
    names = list(KNOWN) + [n for n, _, _ in MEASURED]
    c['map similarity to layer 22, median r (26 peptides)'] = float(np.median([offdiag_r(M[n], M22[n]) for n in names]))
    return c


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if '--rescan' in sys.argv or not os.path.exists(CACHE):
        scan()
    raw = np.load(CACHE)
    allM = {name: sym(raw[name]) for name in raw.files}           # [36, L, L] per peptide
    at = lambda layer: {n: allM[n][layer] for n in allM}
    M22 = at(22)
    rows = {}
    for layer in TOP10:
        M = at(layer)
        rows[layer] = claims(M, M22)
        d = os.path.join(OUT, f'L{layer}')
        os.makedirs(d, exist_ok=True)
        if '--no-figs' not in sys.argv:
            known_figs(M, layer, d)
            write_sets([(n, s, M[n]) for n, s, _ in MEASURED], d, 'helicity', grid=SET_GRID)
        print(f'[layer {layer}] done', flush=True)
    table = pd.DataFrame(rows)
    table.columns = [f'L{c}' for c in table.columns]
    table.round(3).to_csv(os.path.join(OUT, 'claims.csv'))
    with pd.option_context('display.width', 250, 'display.max_colwidth', 48):
        print(table.round(2).to_string())
