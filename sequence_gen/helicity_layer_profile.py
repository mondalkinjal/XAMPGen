"""Correlation between the attention helicity score and the measured helicity,
at every layer of the model.

This is the structural counterpart to the sequence-label probes of Figure 3: it
asks, layer by layer, how well the periodicity of the attention map tracks the
membrane-induced helicity we measured by CD.  Unlike the probes, which plateau
over ten layers that cannot be told apart, this criterion peaks sharply, and the
peak is the layer the attention analysis is carried out at.

Reads the cached head-averaged maps, so it needs no GPU.

    python3 helicity_layer_profile.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
from scipy.stats import pearsonr

from motif_common import sym

# as helicity_scan.PEPTIDES, inlined so this script needs no torch; the third
# field is the CD helicity in lipid of Table 3, as a fraction
PEPTIDES = [
    ("P1", "KWRSFIRGVTHAAHTALKHYFRRKRSG", 1.000),
    ("P2", "LWGHIFRGIKHIGKTIHRLVTG", 0.778),
    ("P3", "RWRSILKHGLKHIGKTIHRLVTG", 0.825),
    ("P4", "VWRHFFRGIRHVGKTIHRLVTG", 0.737),
    ("P5", "VWQHIFRGIVHVGRTIHRLITG", 0.820),
    ("P6", "VTWRHIFRGIVHVGKTIHRLVTG", 0.639),
    ("P7", "VKSWIRVAGRAIGKLIRRK", 0.405),
    ("P8", "IWHHIVRGALHVGKLIHRLVTG", 0.689),
    ("P9 (WT)", "FIHHIIGGLFSAGKAIHRLIRRRRR", 0.521),
    ("P10", "FIHHIIGGLFSAGRAIHRLIRRRRR", 0.599),
    ("P11", "VTWRAIFRGIIHVGKTIHRLVTG", 0.740),
    ("P12", "LWHIFKRIHRGIVHVGKTIHRLVTG", 0.760),
    ("P13", "RWKGVIQAGKAIHRIIRRRR", 0.411),
    ("P14", "VFWAKFKVAFRGVRHIGRTILRHVLR", 0.632),
    ("P15", "VWGFIKHIVRPILRRLTRGLNKL", 0.500),
    ("P16", "NWRNAWRRFKGAFRHVIHTAKRF", 0.440),
    ("P17", "DWRFIRRIWKVGSHIYRAIKKLL", 0.551),
    ("P18", "FIHHIIWILFSAGRAIHRLIRRRRR", 0.667),
]

plt.rcdefaults()

CACHE = os.path.join('top10_layers', 'attn_all_layers.npz')
OUT = os.path.join('..', '69c005365ec28e7b5ef2bc4c', 'Figures')
DMAX, PERIOD = 8, 3.6
PAGE_W = 7.0


def score(M):
    """Mean over residues of the Lomb-Scargle power at PERIOD of the
    coupling-vs-separation profile, d = 1..DMAX.  Index 0 is the AMP token."""
    n, d = M.shape[0] - 1, np.arange(1, DMAX + 1, dtype=float)
    w = np.array([2 * np.pi / PERIOD])
    p = []
    for i in range(1, n + 1):
        prof = np.array([np.mean([M[i, j] for j in (i - k, i + k) if 1 <= j <= n])
                         for k in range(1, DMAX + 1)])
        p.append(np.atleast_1d(lombscargle(d, prof - prof.mean(), w,
                                           normalize=True))[0])
    return np.mean(p)


def main():
    cache = np.load(CACHE)
    y = np.array([h for _, _, h in PEPTIDES])
    nlayers = cache[PEPTIDES[0][0]].shape[0]

    r = np.zeros(nlayers)
    p = np.zeros(nlayers)
    for lay in range(nlayers):
        x = np.array([score(sym(cache[name][lay])) for name, _, _ in PEPTIDES])
        r[lay], p[lay] = pearsonr(x, y)

    best = int(np.argmax(r))
    print(f'{"layer":>5} {"pearson r":>10} {"p":>8}')
    for lay in range(nlayers):
        mark = '  <-- maximum' if lay == best else ''
        print(f'{lay:>5} {r[lay]:>+10.3f} {p[lay]:>8.4f}{mark}')
    print(f'\nbest layer {best}: r = {r[best]:+.3f} (p = {p[best]:.4f})')
    order = np.argsort(-r)
    print('ranked:', ', '.join(f'L{l}({r[l]:+.2f})' for l in order[:5]))

    fig, ax = plt.subplots(figsize=(PAGE_W, 3.4))
    ax.axhline(0, color='0.6', lw=1)
    ax.plot(np.arange(nlayers), r, color='#08519c', lw=1.8,
            marker='o', markersize=4.5, label='all layers')
    ax.plot([best], [r[best]], marker='o', markersize=11, mfc='none',
            mec='#d94801', mew=2.2, zorder=5)
    ax.annotate(f'layer {best}\n$r$ = {r[best]:+.3f}',
                xy=(best, r[best]), xytext=(best + 2.0, r[best] - 0.02),
                fontsize=9, color='#8c2d04',
                arrowprops=dict(arrowstyle='->', color='#d94801', lw=1.3))
    ax.set_xlabel('layer number', fontsize=10)
    ax.set_ylabel('Pearson $r$ with measured\nmembrane helicity', fontsize=10)
    ax.set_xticks(np.arange(0, nlayers, 2))
    ax.tick_params(labelsize=9)
    ax.set_xlim(-0.8, nlayers - 0.2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT, 'helicity_layer_profile.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('wrote', path)


if __name__ == '__main__':
    main()
