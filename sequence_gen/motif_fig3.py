"""The triad figures: every peptide carrying GKA/GKT/GRA/GRT, each shown at the
single (layer, head) that attends to its triad most strongly.

Drawn through motif_common.write_sets with the same print-size grid as
helicity_figs.py, so the panels match the helicity figures exactly: 3x2 to a
figure, rows sized by peptide length, every residue on y, every third on x
unrotated, per-panel colourbar, drawn at the size it is printed.

Writes motif_head_results/triad_panels/triad_set{1,2,3}.{png,pdf}.
No model, no GPU: reads the .npz written by motif_scan.py.

    python3 motif_fig3.py
"""
import numpy as np

from motif_common import SCAN, CACHE, sym, triad_map, short, write_sets

OUTDIR, TAG = 'motif_head_results/triad_panels', 'triad'

# The references first, then every Table 2 peptide carrying the triad.  P9 is
# omitted: it is the same sequence as K1 and selects the same cell.
PANELS = [
    'known|K1 (WT)', 'known|K2', 'known|K3', 'known|K8 (Magainin2)',
    'generated|P2', 'generated|P3',
    'generated|P4', 'generated|P5', 'generated|P6',
    'generated|P7', 'generated|P10', 'generated|P11',
    'generated|P12', 'generated|P13', 'generated|P14', 'generated|P18',
]

# Heatmaps drawn at the size they are printed, so LaTeX includes them at ~1:1
# and the fonts below are the fonts on the page.  Identical to helicity_figs.py.
TEXTWIDTH, TEXTHEIGHT, CAPTION_ROOM = 6.5, 9.0, 0.7
PRINT = dict(
    nrows=3, ncols=2, figsize=(TEXTWIDTH, TEXTHEIGHT - CAPTION_ROOM),
    tick=8, ytick=8, xtick=8, cbtick=10, xstep=3, xrot=0, title_pad=1.5,
    cbar_pad=0.05, tick_pad=1.5, ystep=1, dpi=600, titles_in_gaps=True,
    rows_by_length=True,
    adjust=dict(left=0.06, right=0.94, top=0.985, bottom=0.015,
                hspace=0.06, wspace=0.32),
    rc={'font.size': 6, 'axes.titlesize': 8, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2, 'ytick.major.size': 2},
    pad_inches=0.02)


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
        name = short(str(scan['names'][i]))
        entries.append((f'{name}   L{ly}   H{hd}', seq, sym(A[ly, hd])))
        print(f'  {name:12s} {scan["triads"][i]} {a}-{c} -> L{ly} H{hd}  '
              f'{S[ly, hd]:+.4f}', flush=True)

    write_sets(entries, OUTDIR, TAG, grid=PRINT, formats=('png', 'pdf'))


if __name__ == '__main__':
    main()
