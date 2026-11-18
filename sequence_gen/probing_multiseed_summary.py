"""Mean +/- s.d. of per-layer test metrics over seeds, an overall layer score, and the 3-panel figure."""
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = 'probing_multiseed'
METRICS = [('classification', 'test_mcc', 'MCC'), ('classification', 'test_acc', 'ACC'),
           ('regression', 'R2_test', 'R2'), ('regression', 'PCC_test', 'PCC')]

chunks = [json.load(open(f)) for f in sorted(glob.glob(f'{ROOT}/seed_*/*_L*-*.json'))]
rows = [dict(seed=c['seed'], layer=int(layer), metric=name, value=c[task][layer][key])
        for c in chunks for task, key, name in METRICS if task in c for layer in c[task]]
long = pd.DataFrame(rows)
done = long.groupby('seed')['metric'].size()
runs = [dict(seed=s) for s in done.index if done[s] == 36 * len(METRICS)]
long = long[long.seed.isin([r['seed'] for r in runs])]
names = [m for _, _, m in METRICS]

# overall score per seed: mean over the four metrics of the layer's rank (1 = best)
per_seed = long.pivot_table(index=['seed', 'layer'], columns='metric', values='value')[names]
per_seed['mean_rank'] = per_seed.groupby('seed')[names].rank(ascending=False).mean(axis=1)
best_per_seed = per_seed['mean_rank'].groupby('seed').idxmin().map(lambda t: t[1])

mean = long.pivot_table(index='layer', columns='metric', values='value', aggfunc='mean')[names]
std = long.pivot_table(index='layer', columns='metric', values='value', aggfunc='std')[names]
table = mean.round(4).join(std.round(4), rsuffix='_sd')
table['overall_rank'] = per_seed['mean_rank'].groupby('layer').mean().round(2)
table['overall_rank_sd'] = per_seed['mean_rank'].groupby('layer').std().round(2)
table['times_best'] = best_per_seed.value_counts().reindex(table.index, fill_value=0)
table.to_csv(f'{ROOT}/layer_summary.csv')

print(f'{len(runs)} seeds: {sorted(r["seed"] for r in runs)}')
print('best layer per seed:', dict(best_per_seed))
print(table.sort_values('overall_rank').head(10).to_string())

# 3-panel figure, mean with a +/- s.d. band, same layout as Figure 3
plt.rcParams.update({'font.size': 26, 'axes.labelsize': 30, 'xtick.labelsize': 26,
                     'ytick.labelsize': 26, 'legend.fontsize': 26, 'axes.linewidth': 2.5})
fig, axes = plt.subplots(3, 1, figsize=(16, 20), sharex=True, gridspec_kw={'hspace': 0})
L = mean.index.values
panels = [(axes[0], [('MCC', 'b', 'o'), ('ACC', 'g', 's')], 'Classif.\nScore', 'A'),
          (axes[1], [('R2', 'r', '^')], '$R^2$ Score', 'B'),
          (axes[2], [('PCC', 'm', 'p')], 'PCC Score', 'C')]
for ax, series, ylabel, tag in panels:
    for m, c, mk in series:
        ax.plot(L, mean[m], f'{c}-{mk}', linewidth=4, markersize=12, label=m.replace('ACC', 'Accuracy'))
        ax.fill_between(L, mean[m] - std[m], mean[m] + std[m], color=c, alpha=0.2)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=15)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=2)
    ax.text(0.5, 0.95, tag, transform=ax.transAxes, fontsize=32, fontweight='bold', ha='center', va='top')
axes[0].legend(loc='lower right', frameon=True)
axes[2].set_xlabel('Layer Number', fontweight='bold', labelpad=15)
axes[2].set_xticks([l for l in L if l % 5 == 0])
fig.tight_layout()
fig.savefig(f'{ROOT}/combined_probing_3panel_multiseed.png', dpi=300, bbox_inches='tight')
print(f'figure -> {ROOT}/combined_probing_3panel_multiseed.png')
