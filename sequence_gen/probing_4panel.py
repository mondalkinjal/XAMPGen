"""The three probing tasks in one 4-panel figure, panels stacked vertically:
A-C as in probing_multiseed_summary.py, D the helicity probe of
helicity_layer_profile.py.

Reads the same cached results as those two scripts, so it needs no GPU.

    python3 probing_4panel.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr

from helicity_layer_profile import PEPTIDES

ROOT = 'probing_multiseed'
OUT = os.path.join('..', '69c005365ec28e7b5ef2bc4c', 'Figures')
METRICS = [('classification', 'test_mcc', 'MCC'), ('classification', 'test_acc', 'ACC'),
           ('regression', 'R2_test', 'R2'), ('regression', 'PCC_test', 'PCC')]

# A-C: per-layer probe metrics, mean +/- s.d. over seeds
chunks = [json.load(open(f)) for f in sorted(glob.glob(f'{ROOT}/seed_*/*_L*-*.json'))]
rows = [dict(seed=c['seed'], layer=int(layer), metric=name, value=c[task][layer][key])
        for c in chunks for task, key, name in METRICS if task in c for layer in c[task]]
long = pd.DataFrame(rows)
done = long.groupby('seed')['metric'].size()
long = long[long.seed.isin([s for s in done.index if done[s] == 36 * len(METRICS)])]
names = [m for _, _, m in METRICS]
mean = long.pivot_table(index='layer', columns='metric', values='value', aggfunc='mean')[names]
std = long.pivot_table(index='layer', columns='metric', values='value', aggfunc='std')[names]

# D: correlation of the attention helicity score with the CD helicity, per layer
scores = np.load(os.path.join('top10_layers', 'helicity_scores_all_layers.npy'))
y = np.array([h for _, _, h in PEPTIDES])
r = np.array([pearsonr(scores[lay], y)[0] for lay in range(scores.shape[0])])
best = int(np.argmax(r))
print(f'helicity probe: best layer {best}, r = {r[best]:+.3f}')

plt.rcParams.update({'font.size': 20, 'axes.labelsize': 22, 'xtick.labelsize': 20,
                     'ytick.labelsize': 20, 'legend.fontsize': 19, 'axes.linewidth': 2.0})
fig, axes = plt.subplots(4, 1, figsize=(9, 11.5), sharex=True,
                         gridspec_kw={'hspace': 0.0})
L = mean.index.values
panels = [(axes[0], [('MCC', 'b', 'o'), ('ACC', 'g', 's')], 'Classif.\nScore', 'A'),
          (axes[1], [('R2', 'r', '^')], '$R^2$ Score', 'B'),
          (axes[2], [('PCC', 'm', 'p')], 'PCC Score', 'C')]
for ax, series, ylabel, tag in panels:
    for m, c, mk in series:
        ax.plot(L, mean[m], f'{c}-{mk}', linewidth=3, markersize=8, label=m.replace('ACC', 'Accuracy'))
        ax.fill_between(L, mean[m] - std[m], mean[m] + std[m], color=c, alpha=0.2)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=12)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
    ax.text(0.015, 0.94, tag, transform=ax.transAxes, fontsize=26, fontweight='bold',
            ha='left', va='top')
lo, hi = axes[0].get_ylim()                      # room under the curves for the legend
axes[0].set_ylim(lo - 0.42 * (hi - lo), hi)
axes[0].legend(loc='lower center', ncol=2, frameon=True)

ax = axes[3]
ax.axhline(0, color='0.6', lw=1.5)
ax.plot(np.arange(len(r)), r, color='#08519c', lw=3, marker='o', markersize=8)
ax.plot([best], [r[best]], marker='o', markersize=20, mfc='none', mec='#d94801', mew=3, zorder=5)
lo, hi = ax.get_ylim()                           # room above the peak for its label
ax.set_ylim(lo, hi + 0.32 * (hi - lo))
ax.annotate(f'layer {best}, $r$ = {r[best]:+.3f}', xy=(best, r[best] + 0.04),
            xytext=(best - 13.5, r[best] + 0.26), color='#8c2d04',
            arrowprops=dict(arrowstyle='->', color='#d94801', lw=2.2))
ax.set_ylabel('Helicity $r$', fontweight='bold', labelpad=12)
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
ax.text(0.015, 0.94, 'D', transform=ax.transAxes, fontsize=26, fontweight='bold',
        ha='left', va='top')

axes[3].set_xlabel('Layer Number', fontweight='bold', labelpad=12)
axes[3].set_xticks([l for l in L if l % 5 == 0])
for ax in axes:                                  # panels share a frame: keep the
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))   # y labels off
    ax.tick_params(top=False)                                        # the joins
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
path = os.path.join(OUT, 'combined_probing_4panel.png')
fig.savefig(path, dpi=300, bbox_inches='tight')
print('wrote', path)
