"""Attention figures with the AMP-tag row drawn separately from the residue map.

The reviewers noted that the tag reads as a zeroth amino acid, and that its cell
against the first residue is always strongly negative.  That cell is an artifact
of the baseline of Eq. 1 -- the first residue's row sees only the tag and itself,
so the quantity subtracted there is 1/2, the largest in the map -- and on a shared
colour scale it compresses every residue-residue cell.  Splitting the tag off
fixes both: the strip keeps the tag couplings, and the map below is scaled to the
residue pairs alone.

Every residue is labelled with its identity and position.  Two things pay for
that.  The panels are laid out in absolute inches at the printed size, so a point
here is a point on the page and nothing is rescaled by \\includegraphics; and each
colourbar carries its end values stacked above and below it rather than beside
it, which returns ~0.19 in of width per panel to the x axis.  Where a peptide is
still too long for one label per column -- the 40-mer plectasin and the 37-mer
LL-37, at half page width -- the y axis gives up type size instead of labels
(`y_all`), down to FS_Y_MIN; the x axis still falls back to every second residue.

Writes the main-text figure (Fig. 4) and its SI counterpart, the latter without
the NPR5 panel, which is reserved for a separate paper.

    python3 split_amp_figs.py
"""
import os
import string

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from motif_common import sym

plt.rcdefaults()          # motif_common sizes rcParams for its 60x80 figures

CACHE = os.path.join('probing_results_known_peptides', 'known_L22_cache.npz')
OUT = os.path.join('..', '69c005365ec28e7b5ef2bc4c', 'Figures')

# ---- page and panel geometry, inches ------------------------------------
# PAGE_W is the \textwidth of the A4 geometry in main.tex (21.0 - 2*1.8 cm);
# PAGE_H is what a figure can occupy and still leave its caption on the page.
# The .tex files cap these at 0.92\textheight, which is 9.31 in, so the figures
# print unscaled and the type keeps its stated size.
PAGE_W    = 6.85
PAGE_H    = 9.10   # the default; PAGE_H_BY_FIG overrides it per figure
# M_LEFT and M_RIGHT are floors, not fixed budgets.  They are what the type
# needs at 6.6-7.2 pt, the range the multi-panel figures land in; a figure with
# fewer panels gets larger type, and then the labels are wider than these and
# the margins have to follow them or the labels run off the page.  margins()
# measures the actual glyphs and takes the larger of the two.
M_LEFT    = 0.235  # y tick labels and the 'tag' strip label
M_RIGHT   = 0.345  # the colourbar block: gap, then the end labels' width
M_TOP     = 0.26   # panel title
STRIP_H   = 0.19   # the AMP-tag row, tall enough for its colourbar to read
STRIP_GAP = 0.15   # deliberate gap: the tag is not an amino acid, and the
                   # strip bar's lower end label sits in it
XLAB_H    = 0.30   # floor for the rotated x tick labels and ticks; like the
                   # margins, the real depth is measured at the figure's font
                   # size, since a rotated label is as deep as it is wide
CBAR_W    = 0.09   # the colourbar itself
CBAR_GAP  = 0.025  # clear space between the map and the colourbar block
EDGE_PAD  = 0.02   # the measured bands are exact, which leaves the outermost
                   # glyph sitting on the last pixel row; keep a hair of white
CBAR_PAD  = 0.05
ROW_GAP   = 0.16
COL_GAP   = 0.06
PITCH     = 1.12   # measured: a tick label needs 1.12 x its point size
FS_MIN, FS_MAX = 6.6, 9.0
FS_Y_MIN  = 4.4    # floor for a y axis asked to name every residue of a long
                   # peptide in a panel too short to do it at fs
OVER = M_TOP + STRIP_H + STRIP_GAP + XLAB_H      # the floor; render() measures

# A figure float is one unbreakable box, so an oversized figure cannot move its
# caption to the next page; past a point it pushes the caption into the bottom
# margin and then off the paper.  These are budgets, not heights: \textheight
# is 10.118 in, and each entry is that minus the figure's own caption as
# measured by \setbox, minus 0.25 in for \abovecaptionskip and float
# separation.  A figure that needs less takes less.
PAGE_H_BY_FIG = {
    'known_m_split.png':     8.54,
    'known_s_split.png':     8.94,
    'ordered_set1_split.png': 8.75,
    'ordered_set2_split.png': 8.95,
    'ordered_set3_split.png': 9.10,
    'triad_set1_split.png':   8.15,
    'triad_set2_split.png':   9.10,
    'triad_set3_split.png':   9.10,
    # one full-width panel; None means square maps, page height to suit
    'polyK_difference.png':   None,
}

SEQ = {
    'TP4':         "FIHHIIGGLFSAGKAIHRLIRRRRR",
    'TP4 K-tail':  "FIHHIIGGLFSAGKAIHRLIKKKKK",
    'TP4 no tail': "FIHHIIGGLFSAGKAIHRLI",
    'Plectasin':   "GFGCNGPWDEDDMQCHNHCKSIKGYKGGYCAKGGFVCKCY",
    'Indolicidin': "ILPWKWPWWPWRR",
    'Magainin-2':  "GIGKFLHSAKKFGKAFVGEIMNS",
    'LL-37':       "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
}
LABEL = {'TP4 K-tail': 'TP4-polyK', 'TP4 no tail': 'TP4-noR5'}
SHARED = ('TP4', 'TP4 K-tail', 'TP4 no tail')

FIGS = [('known_m_split.png', ['TP4', 'Magainin-2', 'Plectasin', 'Indolicidin']),
        ('known_s_split.png', ['TP4 K-tail', 'TP4 no tail', 'LL-37'])]


_MEASURE = None


def text_w(s, fs, rotation=0):
    """Size of s at fs points, in inches, as matplotlib will really draw it.

    Returns the width, or for rotation=90 the depth, which is not quite the
    width: rotating carries the font's ascent and descent across with it.


    Not TextPath: that returns the extent of the glyph outlines, which is the
    ink alone and runs 3 to 10 per cent narrower than the advance width the
    text is actually laid out with -- enough to clip a three-character tick
    label against the page edge.  Ask the renderer instead.
    """
    global _MEASURE
    if _MEASURE is None:
        f = plt.figure(figsize=(1, 1), dpi=300)
        _MEASURE = (f, f.canvas.get_renderer())
    f, r = _MEASURE
    t = f.text(0, 0, s, fontsize=fs, rotation=rotation)
    bb = t.get_window_extent(renderer=r)
    t.remove()
    return (bb.height if rotation else bb.width) / 300.0


def margins(fs, labels):
    """Margins and the x-label band that actually hold the type at this size.

    A tick label needs the same room whichever axis it is on: beside the map on
    the left, and below it on the right-angle, where a rotated label is as deep
    as it is wide.  One measurement therefore sizes both.
    """
    tick = fs * 0.35 / 72 + 1.5 / 72               # tick length and its pad
    left = max(text_w(t, fs) for t in labels) + tick + EDGE_PAD
    deep = max(text_w(t, fs, 90) for t in labels) + tick + EDGE_PAD
    num = text_w('-0.000', fs)
    return (max(M_LEFT, left), max(M_RIGHT, CBAR_GAP + num + EDGE_PAD), num,
            max(XLAB_H, deep))


def map_width(ncols, m_left, m_right):
    """Map width in inches for this many columns at these margins."""
    return (PAGE_W - ncols * (m_left + m_right) - (ncols - 1) * COL_GAP) / ncols


def row_heights(rowmax, fs, budget, W, over):
    """Height for each row, and the page height that goes with them.

    Rows are sized one at a time from the longest peptide each one carries,
    rather than all alike.  A row of short peptides does not need the height a
    long one does, and taking it anyway is what forced the long peptides to
    drop every second label.  budget is None for square maps, and otherwise a
    ceiling: if the rows together want more than the caption leaves, they are
    scaled back together and the y axes give up type size instead of labels.
    """
    nrows = len(rowmax)
    fixed = nrows * over + (nrows - 1) * ROW_GAP
    if budget is None:
        heights = [W] * nrows
    else:
        need = [n * PITCH * fs / 72 for n in rowmax]
        k = min(1.0, (budget - fixed) / sum(need))
        heights = [h * k for h in need]
    return heights, sum(heights) + fixed


def step(n, extent, fs):
    """Label every k-th of n residues, k the smallest that avoids collisions."""
    return max(1, int(np.ceil(n * PITCH * fs / (extent * 72))))


def panel(fig, px, ytop, W, H, M, seq, title, rr_lim, fs, page_h, num_w,
          overlay=None, y_all=False):
    """One peptide: the tag strip, a clear gap, the map, then its colourbar.

    overlay, if given, is called with the map axes once it is drawn, for marks
    that belong to one figure rather than to the house style.
    """
    lab = [f'{a}{i}' for i, a in enumerate(seq, 1)]
    n = len(seq)
    sb = ytop - M_TOP - STRIP_H
    mb = sb - STRIP_GAP - H

    def ax(x, y, w, h):
        return fig.add_axes([x / PAGE_W, y / page_h, w / PAGE_W, h / page_h])

    # the tag strip keeps its own scale: the tag is not an amino acid, and its
    # couplings average some two to three times the residue pairs' and would
    # otherwise flatten the map below
    t = M[0, 1:][None, :]
    tl = np.abs(t).max()
    strip = ax(px, sb, W, STRIP_H)
    imt = strip.imshow(t, cmap='PuOr_r', vmin=-tl, vmax=tl, aspect='auto')
    strip.set_yticks([0])
    strip.set_yticklabels(['tag'], fontsize=fs * 0.75)
    strip.set_xticks([])
    for s in strip.spines.values():
        s.set_linewidth(1.2)
    strip.set_title(title, fontsize=fs * 1.30, loc='left', fontweight='bold',
                    pad=fs * 0.55)
    # 'tag' on the axis is terse, so the row is named in full here; the range
    # is read off the strip's own bar
    strip.text(1.0 - 0.04 / W, 1.28, 'AMP tag', transform=strip.transAxes,
               ha='right', va='bottom', fontsize=fs * 0.78, color='0.35')
    # The tag keeps a bar of its own, since it is on a different scale from the
    # map, and that bar is labelled the same way: the two end values stacked
    # above and below it.  The strip is only STRIP_H tall, so the upper label
    # reaches up into the title band and the lower one into STRIP_GAP -- both
    # free at this x, because the title is left-aligned and the grey line above
    # ends at the right edge of the strip, short of the colourbar block.
    cx = px + W + CBAR_GAP + num_w / 2        # centre of the colourbar block
    cbs = ax(cx - CBAR_W / 2, sb, CBAR_W, STRIP_H)
    plt.colorbar(imt, cax=cbs).set_ticks([])
    for sp in cbs.spines.values():
        sp.set_linewidth(0.8)
    # three decimals, as on the map's bar: the two scales are close enough on
    # some peptides that rounding the tag to two places would read as the same
    # number, and the margin is measured from a three-decimal label anyway
    for y, v in ((1.0, tl), (0.0, -tl)):
        cbs.text(0.5, y + (0.11 if y else -0.11), f'{v:.3f}',
                 transform=cbs.transAxes, ha='center',
                 va='bottom' if y else 'top', fontsize=fs)

    mp = ax(px, mb, W, H)
    im = mp.imshow(M[1:, 1:], cmap='RdBu_r', vmin=-rr_lim, vmax=rr_lim,
                   aspect='auto')
    xs, ys = step(n, W, fs), step(n, H, fs)
    # y_all buys every residue a label on a panel that cannot carry them at fs,
    # by setting the y type from the height instead of the other way round.  It
    # is a real cost -- the y axis then reads smaller than the x axis -- so it
    # is off unless a figure asks for it, and it stops at FS_Y_MIN rather than
    # shrink without limit.
    fsy = fs
    if y_all and ys > 1:
        fsy = max(FS_Y_MIN, int(H * 72 / (n * PITCH) * 10) / 10)
        ys = step(n, H, fsy)
    mp.set_xticks(range(0, n, xs))
    mp.set_xticklabels(lab[::xs], rotation=90, fontsize=fs)
    mp.set_yticks(range(0, n, ys))
    mp.set_yticklabels(lab[::ys], fontsize=fsy)
    mp.tick_params(length=fs * 0.35, width=0.9, pad=1.5)
    for s in mp.spines.values():
        s.set_linewidth(1.2)
    mp.grid(False)

    # The map's bar sits in the right margin.  Its end values go above and below
    # it rather than beside it: stacked that way they overhang the bar by about
    # 0.07 in instead of claiming 0.19 in of width, and that width is what the
    # x axis needs to label every residue.
    bh = H * 0.84
    bar = ax(cx - CBAR_W / 2, mb + (H - bh) / 2, CBAR_W, bh)
    cb = fig.colorbar(im, cax=bar)
    cb.set_ticks([])
    for sp in bar.spines.values():
        sp.set_linewidth(0.8)
    # Centred on the bar, at the tick-label size: the margin is measured from
    # these numbers at this font size, so the bar sits in the middle of them and
    # neither runs back over the map nor off the page.
    for y, v in ((1.0, rr_lim), (0.0, -rr_lim)):
        bar.text(0.5, y + (0.035 if y else -0.035), f'{v:.3f}',
                 transform=bar.transAxes, ha='center',
                 va='bottom' if y else 'top', fontsize=fs)
    if overlay is not None:
        overlay(mp, fs)
    return xs, ys, fsy


def render(path, entries, ncols=2, overlay=None, y_all=False):
    """entries: (title, seq, map, colour limit) in reading order."""
    nrows = -(-len(entries) // ncols)
    budget = PAGE_H_BY_FIG.get(os.path.basename(path), PAGE_H)
    nmax = max(len(s) for _, s, _, _ in entries)
    labels = {f'{a}{i}' for _, s, _, _ in entries for i, a in enumerate(s, 1)}
    rowmax = [max(len(entries[i][1])
                  for i in range(r * ncols, min((r + 1) * ncols, len(entries))))
              for r in range(nrows)]

    # The margins depend on the font size, which depends on the map width,
    # which depends on the margins.  Three passes settle it: the floors are
    # already right for the sizes the multi-panel figures reach, so only a
    # figure with larger type moves at all, and then only once.
    m_left, m_right = M_LEFT, M_RIGHT
    for _ in range(3):
        W = map_width(ncols, m_left, m_right)
        fs = min(FS_MAX, max(FS_MIN, int(W * 72 / (PITCH * nmax) * 10) / 10))
        nl, nr, num_w, xlab_h = margins(fs, labels)
        if abs(nl - m_left) < 5e-4 and abs(nr - m_right) < 5e-4:
            break
        m_left, m_right = nl, nr
    over = M_TOP + STRIP_H + STRIP_GAP + xlab_h
    heights, page_h = row_heights(rowmax, fs, budget, W, over)

    fig = plt.figure(figsize=(PAGE_W, page_h))
    notes, ytop = [], page_h
    for i, (title, seq, M, lim) in enumerate(entries):
        if M.shape[0] != len(seq) + 1:
            raise SystemExit(f'{title}: map {M.shape[0]} vs sequence {len(seq)}')
        r, c = divmod(i, ncols)
        if c == 0 and r:
            ytop -= heights[r - 1] + over + ROW_GAP
        px = c * (m_left + W + m_right + COL_GAP) + m_left
        xs, ys, fsy = panel(fig, px, ytop, W, heights[r], M, seq, title, lim,
                            fs, page_h, num_w, overlay, y_all)
        notes.append(f'{title.split()[-1]}:x/{xs},y/{ys}'
                     + ('' if fsy == fs else f'@{fsy:.1f}pt'))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    rows = ' '.join(f'{h:.2f}' for h in heights)
    print(f'wrote {os.path.basename(path)}  {nrows}x{ncols}  page {page_h:.2f} in  '
          f'map W {W:.2f}  rows {rows}  {fs:.1f} pt  [{" ".join(notes)}]')


def main():
    cache = np.load(CACHE)
    M = {k: sym(cache[k]).mean(0) for k in SEQ}
    # one scale for the three TP4 variants so their maps stay comparable
    shared = max(np.abs(M[k][1:, 1:]).max() for k in SHARED)

    for fname, names in FIGS:
        entries = []
        for i, name in enumerate(names):
            lim = shared if name in SHARED else np.abs(M[name][1:, 1:]).max()
            entries.append((f'{string.ascii_uppercase[i]}   {LABEL.get(name, name)}',
                            SEQ[name], M[name], lim))
        render(os.path.join(OUT, fname), entries, y_all=True)


if __name__ == '__main__':
    main()
