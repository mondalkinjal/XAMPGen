"""Shared pieces for the motif / H17 figures: paths, the attention transform,
the four scoring functions, and the panel styling."""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

SCAN = 'motif_head_results/best_head_scan.npz'
CACHE = 'motif_head_results/attn_cache.npz'


def short(name):
    """'K8 (Magainin2)' -> 'K8 Mag2'.  The parentheticals cost 6pt of title."""
    return name.replace(' (WT)', ' WT').replace(' (Magainin2)', ' Mag2')


def sym(A):
    """Drop the uniform row prior 1/(i+1), keep the causal triangle, reflect.
    Affine in A, so it commutes with averaging over heads.  Any leading dims."""
    n = A.shape[-1]
    i = np.arange(n)
    tri = i[:, None] >= i[None, :]
    S = (A - np.where(tri, 1.0 / (i[:, None] + 1.0), 0.0)) * tri
    C = S + S.swapaxes(-1, -2)
    C[..., i, i] /= 2.0
    return C


def triad_map(A, a, c):
    """[36,16] mean of C[m, j] over the three motif rows m and all j != m."""
    rows, L = [a, a + 1, c], A.shape[-1]
    mask = np.ones((3, L), bool)
    for r, m in enumerate(rows):
        mask[r, m] = False
    M = sym(A)[:, :, rows, :]
    return (M * mask).sum((-2, -1)) / mask.sum()


def h17_map(A, h):
    """[36,16] mean of C[h, j] over j != h, the AMP column included."""
    keep = [j for j in range(A.shape[-1]) if j != h]
    return sym(A)[:, :, h, keep].mean(-1)


def anchor_h17(seq):
    """Literal sequence position 17, or None if residue 17 is not a histidine.
    Deliberately does not re-anchor: P3/P6/P11 carry the homologous H at 18."""
    return 17 if len(seq) >= 17 and seq[16] == 'H' else None


TITLE = 56         # panel title
TICK = TITLE       # residue labels, same size as the title

plt.rcParams.update({
    'font.size': TICK, 'axes.titlesize': TITLE, 'axes.labelsize': TICK,
    'xtick.labelsize': TICK, 'ytick.labelsize': TICK, 'legend.fontsize': TICK,
    'axes.titleweight': 'bold', 'axes.linewidth': 2.5, 'figure.titlesize': TITLE,
})
GRID = dict(nrows=3, ncols=2, figsize=(68, 45), title_pad=14,
            adjust=dict(left=0.05, right=0.95, top=0.95, bottom=0.05,
                        hspace=0.26, wspace=0.16))
# Gaps are sized to what must fit in them at TICK=56, no more:
#   hspace -> 2.99 in, holds the rotated x-labels (1.87) + next title (0.78)
#   wspace -> 4.53 in, holds the colorbar numbers (2.33) + next y-labels (1.87)
# Both scale with figsize, so revisit them if TICK or the figure size changes.


def _ticks(lo, hi):
    """At most ~6 round ticks.  Fixed 0.05 steps gave 29 on a full-range panel."""
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if (hi - lo) / step <= 6:
            break
    t = np.arange(np.ceil(lo / step) * step, hi + step / 2, step)
    t = t[(t >= lo) & (t <= hi)]
    return t if len(t) >= 3 else np.linspace(lo, hi, 5)


def _panel(fig, ax, M, title, seq, g=GRID):
    tick = g.get('tick', TICK)
    lo, hi = M.min(), M.max()
    im = ax.imshow(M, cmap='RdBu_r', vmin=lo, vmax=hi, aspect='auto')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=g.get('cbar_pad', 0.15))
    fig.colorbar(im, cax=cax, ticks=_ticks(lo, hi)).ax.yaxis.set_tick_params(
        pad=g.get('tick_pad', 10), labelsize=g.get('cbtick', tick))
    ax.set_title(title, loc='center', fontweight='bold', pad=g['title_pad'])
    lab = ['AMP'] + [f'{aa}{i + 1}' for i, aa in enumerate(seq)]
    # Every residue.  A rotated label needs ~1.2 x TICK of width, so 27 labels
    # at TICK=56 need 25.2 in of axes -- hence figsize width 68, not 36.
    xstep, ystep = g.get('xstep', 1), g.get('ystep', 3)
    ax.set_xticks(np.arange(len(lab))[::xstep])
    ax.set_xticklabels(lab[::xstep], rotation=g.get('xrot', 90),
                       fontsize=g.get('xtick', tick))
    ax.set_yticks(np.arange(len(lab))[::ystep])
    ax.set_yticklabels(lab[::ystep], fontsize=g.get('ytick', tick))
    ax.tick_params(pad=g.get('tick_pad', 3.5))
    ax.grid(False)


def _fit_titles(fig, axes, margin=2.0):
    """Slide each title (rows 1+) sideways into a gap between the x labels of
    the panel above it, nearest the centre.  Returns the vertical overlap in
    pixels still left for titles no gap is wide enough for (0 if none)."""
    r, m = fig.canvas.get_renderer(), margin * fig.dpi / 72
    grid, worst = np.atleast_2d(axes), 0.0
    for i in range(1, grid.shape[0]):
        for low, up in zip(grid[i], grid[i - 1]):
            if not (low.axison and up.axison):
                continue
            low.title.set_x(0.5)
            tb = low.title.get_window_extent(r)
            labs = [t.get_window_extent(r) for t in up.get_xticklabels() if t.get_text()]
            if not labs or tb.y1 + m <= min(b.y0 for b in labs):
                continue  # already clear vertically
            ab = low.get_window_extent(r)
            occ = sorted((b.x0 - m, b.x1 + m) for b in labs)
            edges = [ab.x0] + [e for iv in occ for e in iv] + [ab.x1]
            gaps = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= tb.width]
            if gaps:
                cx = (ab.x0 + ab.x1) / 2
                a, b = min(gaps, key=lambda ab_: abs((ab_[0] + ab_[1]) / 2 - cx))
                low.title.set_x(((a + b) / 2 - ab.x0) / ab.width)
            else:
                worst = max(worst, tb.y1 + m - min(b.y0 for b in labs))
    return worst


def write_sets(entries, outdir, tag, grid=None, formats=('png',)):
    """Chunk 6 to a figure and write <outdir>/<tag>_setN.<fmt>.
    `grid` overrides GRID; its optional 'rc' dict overrides the rcParams
    (font sizes, line widths) for a figure drawn at print size.  With
    'titles_in_gaps', titles dodge the x labels above them sideways, and
    hspace grows only as much as needed for any title that cannot."""
    g = {**GRID, **(grid or {})}
    os.makedirs(outdir, exist_ok=True)
    for k in range(0, len(entries), 6):
        part = entries[k:k + 6]
        with plt.rc_context(g.get('rc', {})):
            kw = {}
            if g.get('rows_by_length'):
                # each row as tall as its longest peptide, so every panel gets
                # the same height per residue and no y labels crowd
                n = [len(seq) + 1 for _, seq, _ in part]
                n += [0] * (g['nrows'] * g['ncols'] - len(n))
                kw['gridspec_kw'] = {'height_ratios': [max(max(r), 1) for r in
                                     np.reshape(n, (g['nrows'], g['ncols']))]}
            fig, axes = plt.subplots(g['nrows'], g['ncols'], figsize=g['figsize'], **kw)
            flat = np.atleast_1d(axes).ravel()
            for ax in flat[len(part):]:
                ax.axis('off')
            for ax, (title, seq, M) in zip(flat, part):
                _panel(fig, ax, M, title, seq, g)
            adjust = dict(g['adjust'])
            fig.subplots_adjust(**adjust)
            if g.get('titles_in_gaps'):
                for _ in range(10):
                    fig.canvas.draw()
                    need = _fit_titles(fig, axes)
                    if need <= 0:
                        break
                    h = np.mean([a.get_window_extent().height for a in flat])
                    adjust['hspace'] += 1.1 * need / h
                    fig.subplots_adjust(**adjust)
                print(f'  hspace {g["adjust"]["hspace"]:.3f} -> {adjust["hspace"]:.3f}', flush=True)
            for fmt in formats:
                out = f'{outdir}/{tag}_set{k // 6 + 1}.{fmt}'
                fig.savefig(out, dpi=g.get('dpi', 300), bbox_inches='tight',
                            pad_inches=g.get('pad_inches', 0.1))
                print(f'  wrote {out}', flush=True)
        plt.close(fig)
