"""The TP4 (polyR) minus TP4-polyK difference map at the cornerstone layer.

One panel: the head-averaged difference between the two effective attention
maps, drawn in the same house style as the other attention figures, with the
difference in AMP-tag coupling as a separate strip above it.

The figure's own point is the dashed box.  The model is causal, so residues
1-20 cannot attend to the tail at 21-25; the upstream block of the difference
is therefore identically zero -- 400 of 400 cells exactly 0.0, not merely
small -- and nothing there can be read as a model finding.  What the
substitution does change is confined to the tail's own rows and their
reflections.

The head-spread numbers that used to be panel B are still printed to stdout,
since the text quotes them.

    python3 polyK_difference.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motif_common import sym
from split_amp_figs import OUT, render

plt.rcdefaults()          # motif_common sizes rcParams for its 60x80 figures

LAYER = 22
TP4 = "FIHHIIGGLFSAGKAIHRLIRRRRR"
TAILS = ['R', 'K', 'A', 'E', 'G', 'S', 'L']
R18, K22 = 18, 22
CORE = 20                 # the last position that cannot see the tail


def band(M):
    """Mean coupling of residue j to every other residue; any leading dims."""
    n = M.shape[-1]
    out = np.zeros(M.shape[:-2] + (n,))
    for j in range(n):
        sel = [k for k in range(n) if k != j]
        out[..., j] = M[..., sel, j].mean(-1)
    return out


def mark_core(mp, fs):
    """Box the upstream block and say why it is blank."""
    mp.add_patch(plt.Rectangle((-0.5, -0.5), CORE, CORE, fill=False,
                               ec='0.30', lw=1.1, ls=(0, (4, 3))))
    mp.text(CORE / 2 - 0.5, CORE / 2 - 0.5,
            'identically zero\n(causal mask)',
            ha='center', va='center', fontsize=fs * 1.15, color='0.30',
            linespacing=1.4)


def main():
    d = np.load(os.path.join('top10_layers', 'tail_swap.npz'))
    M = {t: sym(d[f'TP4|{t}'][LAYER]) for t in TAILS}      # [16, 26, 26] per tail
    D = M['R'].mean(0) - M['K'].mean(0)

    lim = np.abs(D[1:, 1:]).max()
    render(os.path.join(OUT, 'polyK_difference.png'),
           [('TP4 (polyR) $-$ TP4-polyK, layer 22', TP4, D, lim)],
           ncols=1, overlay=mark_core)

    # --- numbers quoted in the text ----------------------------------------
    print('upstream block: max |difference| = %.1e over %d cells, %d exactly zero'
          % (np.abs(D[1:CORE + 1, 1:CORE + 1]).max(), CORE * CORE,
             int((D[1:CORE + 1, 1:CORE + 1] == 0).sum())))
    per_head = M['R'][:, R18, K22] - M['K'][:, R18, K22]
    print('%s%d--position-%d change: head mean %+.4f, sd over heads %.4f, max head %+.4f'
          % (TP4[R18 - 1], R18, K22, per_head.mean(), per_head.std(ddof=1),
             per_head.max()))
    for t in TAILS:
        b = band(M[t].mean(0))
        bh = band(M[t])[:, K22]
        rank = int(np.argsort(np.argsort(-b[1:26]))[K22 - 1]) + 1
        print(f'{t}: band({K22}) = {b[K22]:+.4f}  sd over heads = {bh.std(ddof=1):.4f}'
              f'  rank {rank}/25'
              + ('' if t == 'R' else
                 '  d(R18-22) = %+.4f' % (M['R'].mean(0) - M[t].mean(0))[R18, K22]))


if __name__ == '__main__':
    main()
