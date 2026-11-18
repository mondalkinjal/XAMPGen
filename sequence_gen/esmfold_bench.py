"""ESMFold (Lin et al. 2023) vs measured helicity -- an MSA-free structure
predictor, so a predicted helix FRACTION rather than a propensity.

ESMFold folds from a single sequence (ESM-2 3B language model + folding trunk),
which is what makes it usable here: 19-27-residue AMPs have no usable MSA, so
AlphaFold2-style predictors are out of their regime.

Helix assignment: no DSSP binary on this cluster, so secondary structure is
assigned from backbone dihedrals.  Two Ramachandran boxes are reported, a
standard one and a permissive one, to show the number is not an artefact of
where the cutoff is drawn.

Readouts per peptide:
    helix fraction (strict / loose phi-psi box)
    mean pLDDT                 -- confidence, a proxy for how ordered it thinks
                                  the peptide is
    mean pLDDT over helical residues

Run on a compute node with the HF cache pre-populated:
    HF_HOME=.../hf_cache HF_HUB_OFFLINE=1 python3 esmfold_bench.py
"""
from __future__ import print_function

import os
import re
import numpy as np
import torch

OUT = 'esmfold_out'
MODEL = 'facebook/esmfold_v1'

PEPTIDES = [(n, q, float(v)) for n, q, v in re.findall(
    r'\("([^"]+)",\s*"([A-Z]+)",\s*([\d.]+)\)',
    open('helicity_scan.py').read())]

# phi, psi windows for alpha-helix.  Ideal alpha is about (-57, -47).
BOX_STRICT = ((-100.0, -30.0), (-80.0, -5.0))
BOX_LOOSE = ((-160.0, -20.0), (-120.0, 30.0))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def backbone_from_pdb(pdb):
    """N, CA, C coordinates and per-residue B-factor (pLDDT) in order."""
    res = {}
    for ln in pdb.splitlines():
        if not ln.startswith('ATOM'):
            continue
        name = ln[12:16].strip()
        if name not in ('N', 'CA', 'C'):
            continue
        ri = int(ln[22:26])
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        res.setdefault(ri, {})[name] = xyz
        if name == 'CA':
            res[ri]['b'] = float(ln[60:66])
    ks = sorted(res)
    return [res[k] for k in ks if {'N', 'CA', 'C'} <= set(res[k])]


def helix_flags(bb, box):
    """True where (phi, psi) falls in `box`.  Termini have no phi/psi and are
    counted as non-helical, the same convention DSSP effectively uses."""
    (plo, phi_), (slo, shi) = box
    n = len(bb)
    out = np.zeros(n, bool)
    for i in range(1, n - 1):
        ph = dihedral(bb[i - 1]['C'], bb[i]['N'], bb[i]['CA'], bb[i]['C'])
        ps = dihedral(bb[i]['N'], bb[i]['CA'], bb[i]['C'], bb[i + 1]['N'])
        out[i] = (plo <= ph <= phi_) and (slo <= ps <= shi)
    return out


def main():
    from transformers import AutoTokenizer, EsmForProteinFolding
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = EsmForProteinFolding.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    model.trunk.set_chunk_size(64)
    print('[esmfold] loaded', flush=True)

    rows = []
    for name, seq, h in PEPTIDES:
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        pdb_path = '%s/%s.pdb' % (OUT, safe)
        if os.path.exists(pdb_path):
            pdb = open(pdb_path).read()
        else:
            with torch.no_grad():
                pdb = model.infer_pdb(seq)
            open(pdb_path, 'w').write(pdb)
        bb = backbone_from_pdb(pdb)
        assert len(bb) == len(seq), '%s: %d backbone vs %d residues' % (
            name, len(bb), len(seq))
        hs = helix_flags(bb, BOX_STRICT)
        hl = helix_flags(bb, BOX_LOOSE)
        plddt = np.array([r['b'] for r in bb])
        rows.append(dict(name=name, seq=seq, y=float(h), n=len(seq),
                         fs=hs.mean(), fl=hl.mean(),
                         plddt=plddt.mean(),
                         plddt_h=plddt[hs].mean() if hs.any() else np.nan,
                         ss=''.join('H' if a else '-' for a in hs)))
        print('  %-10s meas %.3f  helixfrac %.3f  pLDDT %.1f' %
              (name, h, hs.mean(), plddt.mean()), flush=True)

    from scipy.stats import pearsonr, spearmanr
    y = np.array([r['y'] for r in rows])
    out = []
    w = out.append
    w('=' * 94)
    w(' ESMFold (MSA-free) vs measured helicity, N = %d' % len(rows))
    w(' helix assigned from backbone phi/psi (no DSSP available); pLDDT = B-factor')
    w('=' * 94)
    w('')
    w('%-10s %3s %8s %9s %9s %7s   %s' %
      ('peptide', 'n', 'measured', 'helixfrac', 'loose', 'pLDDT', 'phi/psi helix'))
    w('-' * 94)
    for r in sorted(rows, key=lambda r: -r['y']):
        w('%-10s %3d %8.3f %9.3f %9.3f %7.1f   %s' %
          (r['name'], r['n'], r['y'], r['fs'], r['fl'], r['plddt'], r['ss']))
    w('')
    w('=' * 94)
    for key, lab in (('fs', 'ESMFold helix fraction (strict box)'),
                     ('fl', 'ESMFold helix fraction (loose box)'),
                     ('plddt', 'ESMFold mean pLDDT'),
                     ('plddt_h', 'ESMFold mean pLDDT, helical residues')):
        x = np.array([r[key] for r in rows])
        ok = np.isfinite(x)
        if ok.sum() < 4 or x[ok].std() == 0:
            w('%-38s  constant or undefined (sd = 0)' % lab)
            continue
        r_, p_ = pearsonr(x[ok], y[ok])
        rho, pr = spearmanr(x[ok], y[ok])
        w('%-38s  r = %+.3f (p = %.4f)   rho = %+.3f (p = %.4f)'
          % (lab, r_, p_, rho, pr))
    x = np.array([r['fs'] for r in rows])
    w('')
    w('calibration of helix fraction as a direct helicity predictor:')
    w('  measured   mean %.3f  sd %.3f  range %.3f-%.3f'
      % (y.mean(), y.std(ddof=1), y.min(), y.max()))
    w('  ESMFold    mean %.3f  sd %.3f  range %.3f-%.3f'
      % (x.mean(), x.std(ddof=1), x.min(), x.max()))
    w('  RMSE = %.3f   mean signed error = %+.3f'
      % (np.sqrt(np.mean((x - y) ** 2)), (x - y).mean()))

    txt = '\n'.join(out)
    print('\n' + txt)
    open('esmfold_bench_results.txt', 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
