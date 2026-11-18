"""LaTeX rows for the two alignment tables of the SI, with the shared residues bold.

Reads the two BLAST outputs written by blast_align.sh and emits, for each, one
row per peptide: the peptide's own sequence and the sequence it aligns to, with
\textbf{} around every residue the alignment matches identically.

    python3 align_tables.py train   > rows_train.tex    # Table S2
    python3 align_tables.py sp      > rows_sp.tex       # Table S3

Identity is identical residues over the FULL peptide length (nident/qlen), not
over the alignment length: blastp-short is local and will report 100% over a
seven-residue window.  The best hit is the highest identity, ties broken by the
lower E value.
"""
import json
import re
import sys

G = (str(__import__('pathlib').Path(__file__).resolve().parents[1]) + '/')
ORIGIN = {'P9_WT': 'wild type', 'P10': 'human', 'P18': 'human'}
GROUPS = [['P9_WT', 'P10', 'P18'],
          ['P6', 'P2', 'P4', 'P5', 'P11', 'P8', 'P12'],
          ['P13', 'P3'],
          ['P17', 'P7', 'P16', 'P14', 'P1', 'P15']]
# Swiss-Prot titles are long; these are the names used in the table
SPNAME = {'P9_WT': 'Pteroicidin-$\\alpha$', 'P10': 'Pteroicidin-$\\alpha$',
          'P18': 'Pteroicidin-$\\alpha$', 'P6': 'Moronecidin', 'P2': 'Moronecidin',
          'P4': 'Moronecidin', 'P5': 'Moronecidin', 'P11': 'Moronecidin',
          'P8': 'Moronecidin', 'P12': 'Moronecidin', 'P13': 'Chrysophsin-1',
          'P3': 'Mediator of RNA pol.\\ II subunit', 'P17': 'Leucine--tRNA ligase',
          'P7': 'Pyruvate aldolase', 'P16': 'HMG-CoA synthase',
          'P14': 'dGTP triphosphohydrolase', 'P1': 'LRR receptor-like kinase',
          'P15': 'Asparagine--tRNA ligase'}


def fasta(path):
    d, n = {}, None
    for l in open(path):
        l = l.strip()
        if l.startswith('>'):
            n = l[1:]
            d[n] = ''
        elif n:
            d[n] += l
    return d


def best_hits(path, pep, exclude_self=None):
    """One hit per peptide, from `-outfmt '6 qseqid sseqid qlen slen qstart qend
    sstart send evalue nident qseq sseq [stitle]'`."""
    best = {}
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        q, sid, qlen, ev, nid = f[0], f[1], int(f[2]), float(f[8]), int(f[9])
        if exclude_self and exclude_self.get(sid) == pep[q]:
            continue                      # P9 is the TP4 wild type: it IS a training entry
        ident = 100 * nid / qlen
        if q not in best or (ident, -ev) > (best[q]['ident'], -best[q]['ev']):
            best[q] = dict(ident=ident, ev=ev, sid=sid, slen=int(f[3]),
                           qs=int(f[4]), qe=int(f[5]), ss=int(f[6]), se=int(f[7]),
                           qseq=f[10], sseq=f[11],
                           title=f[12] if len(f) > 12 else '',
                           # '*' marks a column where both residues are the same
                           mark=''.join('*' if a == b and a != '-' else ' '
                                        for a, b in zip(f[10], f[11])))
    return best


def runs(chars, flags):
    """\textbf{} around maximal runs of matched residues, so the bold blocks read
    as blocks rather than as a stutter of one-residue groups."""
    out, i = [], 0
    while i < len(chars):
        j = i
        while j < len(chars) and flags[j] == flags[i]:
            j += 1
        seg = ''.join(chars[i:j])
        out.append('\\textbf{%s}' % seg if flags[i] else seg)
        i = j
    return ''.join(out)


def bold_full(seq, start, aln, mark):
    """The whole sequence, with the aligned-and-identical residues bold.
    `start` is the 1-based offset of the alignment in `seq`; gaps are skipped."""
    flags = [False] * len(seq)
    pos = start - 1
    for c, m in zip(aln, mark):
        if c == '-':
            continue
        flags[pos] = (m == '*')
        pos += 1
    return runs(list(seq), flags)


def bold_window(aln, mark):
    """Just the aligned window, gaps closed up."""
    chars = [c for c in aln if c != '-']
    flags = [m == '*' for c, m in zip(aln, mark) if c != '-']
    return runs(chars, flags)


def evalue(x):
    if x >= 0.1:
        return '%.3g' % x
    e = int(('%e' % x).split('e')[1])
    m = round(x / 10 ** e)
    if m == 10:
        m, e = 1, e + 1
    return '$%d\\times10^{%d}$' % (m, e)


def main(which):
    pep = fasta(G + 'helicity_18.fasta')
    train = {'amp%d' % i: l.strip() for i, l
             in enumerate(open(G + 'unique_combined_sequences.txt'), 1)}

    if which == 'train':
        best = best_hits('aln_train.tsv', pep, exclude_self=train)
    else:
        best = best_hits('aln_sp.tsv', pep)

    for gi, group in enumerate(GROUPS):
        for k in group:
            d = best[k]
            name = 'P9' if k == 'P9_WT' else k
            q = bold_full(pep[k], d['qs'], d['qseq'], d['mark'])
            if which == 'train':
                # a mature peptide of comparable length is shown whole; a
                # precursor only over the aligned segment, marked with a dagger
                if d['slen'] > 40:
                    s = bold_window(d['sseq'], d['mark']) + '$^\\dagger$ (%d)' % d['slen']
                else:
                    s = bold_full(train[d['sid']], d['ss'], d['sseq'], d['mark'])
                print('    %-4s& %-10s& \\texttt{%s} & \\texttt{%s} & %.1f & %s \\\\'
                      % (name, ORIGIN.get(k, 'generated'), q, s, d['ident'],
                         evalue(d['ev'])))
            else:
                s = bold_window(d['sseq'], d['mark'])
                print('    %-4s& \\texttt{%s} & \\texttt{%s} & %s (%d--%d of %d) & %.1f & %s \\\\'
                      % (name, q, s, SPNAME[k], d['ss'], d['se'], d['slen'],
                         d['ident'], evalue(d['ev'])))
        if gi < len(GROUPS) - 1:
            print('    \\midrule')


if __name__ == '__main__':
    main(sys.argv[1])
