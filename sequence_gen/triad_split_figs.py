"""The triad figures, redrawn with the AMP tag split off the residue map.

Same content and same (layer, head) selection as motif_fig3.py -- every peptide
carrying GKA/GKT/GRA/GRT, each at the single head that attends to its triad most
strongly -- but drawn through split_amp_figs.panel so these panels match the rest
of the attention figures: the tag on its own strip and its own colour scale, with
the residue-residue map scaled to the residue pairs alone.

No model, no GPU: reads the .npz written by motif_scan.py.

    python3 triad_split_figs.py
"""
import os
import string

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from motif_common import SCAN, CACHE, sym, triad_map, short
from split_amp_figs import render

plt.rcdefaults()

OUT = os.path.join('..', '69c005365ec28e7b5ef2bc4c', 'Figures')

# as motif_fig3.PANELS; P9 is omitted, being the same sequence as K1
PANELS = [
    'known|K1 (WT)', 'known|K2', 'known|K3', 'known|K8 (Magainin2)',
    'generated|P2', 'generated|P3',
    'generated|P4', 'generated|P5', 'generated|P6',
    'generated|P7', 'generated|P10', 'generated|P11',
    'generated|P12', 'generated|P13', 'generated|P14', 'generated|P18',
]
SETS = [('triad_set1_split.png', 0, 6),
        ('triad_set2_split.png', 6, 12),
        ('triad_set3_split.png', 12, 16)]


def main():
    scan = np.load(SCAN, allow_pickle=True)
    cache = np.load(CACHE)
    keyed = {f"{scan['panels'][i]}|{name}": i
             for i, name in enumerate(map(str, scan['names']))}

    entries = []
    for key in PANELS:
        i = keyed[key]
        seq = str(scan['seqs'][i])
        a, c = map(int, scan['pos'][i])
        A = cache[key]
        S = triad_map(A, a, c)
        ly, hd = map(int, np.unravel_index(S.argmax(), S.shape))
        entries.append((f'{short(str(scan["names"][i]))}   L{ly} H{hd}',
                        seq, sym(A[ly, hd])))
        print(f'  {short(str(scan["names"][i])):12s} -> L{ly} H{hd}', flush=True)

    for fname, lo, hi in SETS:
        chunk = [(f'{string.ascii_uppercase[i]}   {t}', seq, M,
                  np.abs(M[1:, 1:]).max())
                 for i, (t, seq, M) in enumerate(entries[lo:hi])]
        render(os.path.join(OUT, fname), chunk)


if __name__ == '__main__':
    main()
