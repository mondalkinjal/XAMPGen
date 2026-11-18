"""Helix fraction of predicted structures by DSSP, for any MSA-free predictor.

Replaces the phi/psi Ramachandran boxes of fold_helix_bench.py with the same
hydrogen-bond assignment used for the external PDB set in
external_set/dssp_run.py: pydssp on N/CA/C/O, out_type='c3', helix fraction =
fraction of residues assigned 'H'.  Scoring is otherwise identical, so the two
predictors stay comparable with each other and with the external set.

    python3 fold_helix_dssp.py esmfold_out "ESMFold"
    python3 fold_helix_dssp.py omegafold_out "OmegaFold"

Needs pydssp, which lives in the sequence_generation_new conda env:

    env -u PYTHONPATH .../envs/sequence_generation_new/bin/python3 fold_helix_dssp.py ...
"""
from __future__ import print_function

import glob
import os
import re
import sys
import numpy as np
import pydssp
from scipy.stats import pearsonr, spearmanr

PEPTIDES = [(n, q, float(v)) for n, q, v in re.findall(
    r'\("([^"]+)",\s*"([A-Z]+)",\s*([\d.]+)\)',
    open('helicity_scan.py').read())]

WANT = ['N', 'CA', 'C', 'O']


def backbone(pdb):
    """N, CA, C, O coordinates and the CA B-factor (pLDDT), in residue order."""
    res = {}
    for ln in pdb.splitlines():
        if not ln.startswith('ATOM'):
            continue
        nm = ln[12:16].strip()
        if nm not in WANT:
            continue
        ri = int(ln[22:26])
        res.setdefault(ri, {})[nm] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        if nm == 'CA':
            try:
                res[ri]['b'] = float(ln[60:66])
            except ValueError:
                res[ri]['b'] = np.nan
    keys = [k for k in sorted(res) if all(a in res[k] for a in WANT)]
    xyz = np.array([[res[k][a] for a in WANT] for k in keys], dtype=np.float32)
    b = np.array([res[k].get('b', np.nan) for k in keys], dtype=float)
    return xyz, b


def find_pdb(d, name):
    safe = name.replace(' ', '_').replace('(', '').replace(')', '')
    for cand in (safe, name, safe.split('_')[0]):
        p = os.path.join(d, cand + '.pdb')
        if os.path.exists(p):
            return p
    hits = [p for p in glob.glob(os.path.join(d, '*.pdb'))
            if os.path.basename(p).split('.')[0] in (safe, name, safe.split('_')[0])]
    return hits[0] if hits else None


def main(d, label):
    rows = []
    for name, seq, h in PEPTIDES:
        p = find_pdb(d, name)
        if p is None:
            print('  [skip] no PDB for %s in %s' % (name, d))
            continue
        xyz, b = backbone(open(p).read())
        if len(xyz) != len(seq):
            print('  [warn] %s: %d complete backbone residues vs %d in sequence'
                  % (name, len(xyz), len(seq)))
        ss = np.asarray(list(pydssp.assign(xyz, out_type='c3')))
        hel = ss == 'H'
        if np.nanmax(b) > 1.5:          # pLDDT written on 0-100 by some predictors
            b = b / 100.0
        rows.append(dict(name=name, y=h, n=len(seq), f=float(hel.mean()),
                         plddt=np.nanmean(b),
                         plddt_h=np.nanmean(b[hel]) if hel.any() else np.nan,
                         ss=''.join(ss)))

    y = np.array([r['y'] for r in rows])
    out = []
    w = out.append
    w('=' * 94)
    w(' %s (MSA-free) vs measured helicity, N = %d  --  DSSP (pydssp, c3)' % (label, len(rows)))
    w('=' * 94)
    w('%-10s %3s %8s %9s %7s   %s' %
      ('peptide', 'n', 'measured', 'helixfrac', 'pLDDT', 'DSSP'))
    w('-' * 94)
    for r in sorted(rows, key=lambda r: -r['y']):
        w('%-10s %3d %8.3f %9.3f %7.3f   %s' %
          (r['name'], r['n'], r['y'], r['f'], r['plddt'], r['ss']))
    w('')
    for key, lab in (('f', '%s helix fraction (DSSP)' % label),
                     ('plddt', '%s mean pLDDT' % label),
                     ('plddt_h', '%s pLDDT, helical residues' % label)):
        x = np.array([r[key] for r in rows], float)
        ok = np.isfinite(x)
        if ok.sum() < 4 or x[ok].std() == 0:
            w('%-40s  constant / undefined (sd = 0)' % lab)
            continue
        w('%-40s  r = %+.3f (p = %.4f)   rho = %+.3f (p = %.4f)'
          % (lab, *pearsonr(x[ok], y[ok]), *spearmanr(x[ok], y[ok])))
    x = np.array([r['f'] for r in rows])
    w('')
    w('saturation check (can it RANK these peptides?)')
    w('  measured  mean %.3f  sd %.3f  range %.3f-%.3f'
      % (y.mean(), y.std(ddof=1), y.min(), y.max()))
    w('  %-9s mean %.3f  sd %.3f  range %.3f-%.3f'
      % (label, x.mean(), x.std(ddof=1), x.min(), x.max()))
    w('  RMSE vs measured = %.3f   mean signed error = %+.3f'
      % (np.sqrt(np.mean((x - y) ** 2)), (x - y).mean()))

    txt = '\n'.join(out)
    print(txt)
    tag = label.lower().replace(' ', '_')
    open('%s_helix_dssp_results.txt' % tag, 'w').write(txt + '\n')
    np.savez('%s_helix_dssp.npz' % tag,
             names=np.array([r['name'] for r in rows]),
             y=y, f=x, plddt=np.array([r['plddt'] for r in rows], float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
