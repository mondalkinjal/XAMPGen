"""The four figure sets: triad and H17, each at its best (layer, head) and at
its best layer with the 16 heads averaged.  Panels are 3x2, six to a figure.

    python3 motif_figs.py triad_cell  triad_layer  h17_cell  h17_layer
    python3 motif_figs.py all

Reads the .npz written by motif_scan.py; no forward pass, no GPU.
"""
import sys
import numpy as np
from motif_common import (SCAN, CACHE, sym, triad_map, h17_map, anchor_h17,
                          short, write_sets)

OUT = 'motif_head_results/pooled_panels'
SETS = ('triad_cell', 'triad_layer', 'h17_cell', 'h17_layer')


def build(which):
    scan = np.load(SCAN, allow_pickle=True)
    cache = np.load(CACHE)
    entries, seen = [], set()

    for i, name in enumerate(map(str, scan['names'])):
        seq, a, c = str(scan['seqs'][i]), *map(int, scan['pos'][i])
        A = cache[f"{scan['panels'][i]}|{name}"]

        if which.startswith('h17'):
            if seq in seen:
                print(f'  skip duplicate: {name}'); continue
            h = anchor_h17(seq)
            if h is None:
                print(f'  skip, residue 17 = {seq[16] if len(seq) > 16 else "-"}: {name}')
                continue
            seen.add(seq)
            S, what = h17_map(A, h), f'H{h}'
        else:
            S, what = triad_map(A, a, c), f"{scan['triads'][i]}{a}"

        if which.endswith('cell'):
            ly, hd = np.unravel_index(S.argmax(), S.shape)
            M, score, where = A[ly, hd], S[ly, hd], f'L{ly}   H{hd}'
        else:
            p = S.mean(1)                                # average the 16 heads
            ly = int(p.argmax())
            M, score, where = A[ly].mean(0), p[ly], f'L{ly}'

        entries.append((f'{short(name)}   {where}', seq, sym(M)))
        print(f'  {name:16s} {what:12s} -> {where:18s} {score:+.4f}', flush=True)

    return entries


if __name__ == '__main__':
    args = sys.argv[1:] or ['all']
    for which in (SETS if args == ['all'] else args):
        assert which in SETS, f'unknown set {which}; pick from {SETS}'
        print(f'\n==== {which}', flush=True)
        write_sets(build(which), f'{OUT}/{which}', which)
