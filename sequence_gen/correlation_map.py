"""Correlation map of the measured and calculated parameters of Table 3.

Rebuilt from the table values so the figure is reproducible and its labels are
typeset properly (I_max, k_D, muH and the linear moment all carry subscripts).
Spearman is used throughout: MIC is measured on a doubling dilution series, so
only the ordering of the values is meaningful.

    python3 correlation_map.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = os.path.join('..', '69c005365ec28e7b5ef2bc4c', 'Figures')

# Name, MIC E.coli, MIC S.epid, length, %helicity buffer, %helicity lipid,
# n, k_D, I_max, muH, H, linear moment, net charge        (Table 3)
TABLE = """P1  1  0.5 27  0.00 100.0 13.23  0.47 5.96 0.51 0.76 0.27 8
P2  1  1   22  0.00  77.8  7.55  0.12 6.72 0.88 0.46 0.24 4
P3  1  0.5 23  0.00  82.5  9.35  1.07 6.09 0.84 0.68 0.25 6
P4  2  1   22  0.00  73.7  6.14  3.85 7.05 0.87 0.48 0.25 5
P5  2  1   22  4.70  82.0  3.50  2.62 7.05 0.76 0.28 0.25 3
P6  2  0.5 23  1.50  63.9  3.35  7.28 6.41 0.74 0.40 0.21 4
P7  2  0.5 19  0.00  40.5  9.44  9.10 7.80 0.55 0.62 0.25 7
P8  2  1   22  5.20  68.9  3.96  0.72 5.90 0.71 0.43 0.23 3
P9  4  1   25  0.00  52.1   nan   nan  nan 0.64 0.56 0.23 7
P10 8  1   25  0.00  59.9   nan   nan  nan 0.61 0.52 0.23 7
P11 4  0.5 23  4.50  74.0  3.97  2.07 7.50 0.67 0.29 0.24 4
P12 4  0.5 25  2.60  76.0  7.17  1.03 6.32 0.81 0.49 0.27 5
P13 8  1   20  0.00  41.1  6.44 15.17 7.61 0.43 0.79 0.21 8
P14 8  1   26  1.50  63.2  8.18  4.08 5.14 0.61 0.35 0.19 7
P15 8  1   23  3.20  50.0  6.29  2.84 6.90 0.72 0.24 0.22 6
P16 8  0.5 23  0.20  44.0  5.73  3.08 4.93 0.85 0.57 0.23 7
P17 16 2   23  9.10  55.1  1.00 19.50 3.18 0.69 0.42 0.19 6
P18 64 64  25 26.20  66.7  1.00 22.75 2.72 0.50 0.30 0.30 7"""

LABELS = [r'MIC$_{E.\,coli}$', r'MIC$_{S.\,epid.}$', 'Length',
          r'% helicity$_{\mathrm{buffer}}$', r'% helicity$_{\mathrm{lipid}}$',
          r'$n$', r'$k_D$', r'$I_{\max}$', r'$\mu$H', r'$H$',
          r'$\mu$H$_{\mathrm{lin}}$', 'Net charge']


def main():
    rows = [r.split() for r in TABLE.split('\n')]
    A = np.array([[float(x) for x in r[1:]] for r in rows])
    k = A.shape[1]

    C = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            m = ~(np.isnan(A[:, i]) | np.isnan(A[:, j]))
            if m.sum() > 2:
                C[i, j] = spearmanr(A[m, i], A[m, j]).statistic

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels(LABELS, fontsize=9)
    for i in range(k):
        for j in range(k):
            if not np.isnan(C[i, j]):
                ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if abs(C[i, j]) > 0.6 else '0.15')
    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('Spearman rank correlation', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.tight_layout()
    path = os.path.join(OUT, 'experiment_main_fig_relabelled.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print('wrote', path)


if __name__ == '__main__':
    main()
