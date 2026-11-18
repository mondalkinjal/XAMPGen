"""The one forward pass: run each motif-bearing peptide through the finetuned
model and cache its [36,16,L,L] attention stack.

Writes motif_head_results/{attn_cache.npz, best_head_scan.npz}; everything else
in the pipeline reads those and needs no model.  CPU is fine -- load_model()
falls back on its own, ~15 min for the panel.

    python3 motif_scan.py
"""
import numpy as np
import torch

import motif_head_lib as L

TRIADS = ('GKA', 'GKT', 'GRA', 'GRT')

# the motif-bearing peptides, in the order the figure panels use
GENERATED = [
    ("P2",  "LWGHIFRGIKHIGKTIHRLVTG"),
    ("P3",  "RWRSILKHGLKHIGKTIHRLVTG"),
    ("P4",  "VWRHFFRGIRHVGKTIHRLVTG"),
    ("P5",  "VWQHIFRGIVHVGRTIHRLITG"),
    ("P6",  "VTWRHIFRGIVHVGKTIHRLVTG"),
    ("P7",  "VKSWIRVAGRAIGKLIRRK"),
    ("P9 (WT)", "FIHHIIGGLFSAGKAIHRLIRRRRR"),
    ("P10", "FIHHIIGGLFSAGRAIHRLIRRRRR"),
    ("P11", "VTWRAIFRGIIHVGKTIHRLVTG"),
    ("P12", "LWHIFKRIHRGIVHVGKTIHRLVTG"),
    ("P13", "RWKGVIQAGKAIHRIIRRRR"),
    ("P14", "VFWAKFKVAFRGVRHIGRTILRHVLR"),
    ("P18", "FIHHIIWILFSAGRAIHRLIRRRRR"),
]
KNOWN = [
    ("K1 (WT)", "FIHHIIGGLFSAGKAIHRLIRRRRR"),
    ("K2", "FIHHIIGGLFSAGKAIHRLIKKKKK"),
    ("K3", "FIHHIIGGLFSAGKAIHRLI"),
    ("K8 (Magainin2)", "GIGKFLHSAKKFGKAFVGEIMNS"),
]
PANELS = [("generated", GENERATED), ("known", KNOWN)]


def main():
    model, dev = L.load_model()
    names, panels, seqs, triads, pos, cache = [], [], [], [], [], {}

    for panel, entries in PANELS:
        for name, seq in entries:
            hits = [(seq[p:p + 3], p) for p in range(len(seq) - 2)
                    if seq[p:p + 3] in TRIADS]
            if not hits:
                continue
            with torch.no_grad():
                _, attn = L.run(model, torch.tensor([L.encode(seq)], device=dev))
            cache[f'{panel}|{name}'] = torch.stack(attn, 0)[:, 0].float().cpu().numpy()
            for t, p in hits:
                names.append(name); panels.append(panel); seqs.append(seq)
                triads.append(t); pos.append((p + 1, p + 3))
                print(f'[scan] {name:15s} {t} {p + 1}-{p + 3}', flush=True)

    np.savez_compressed('motif_head_results/best_head_scan.npz',
                        names=np.array(names), panels=np.array(panels),
                        seqs=np.array(seqs), triads=np.array(triads),
                        pos=np.array(pos))
    np.savez_compressed('motif_head_results/attn_cache.npz', **cache)
    print(f'\n[done] {len(names)} occurrences, {len(cache)} sequences cached', flush=True)


if __name__ == '__main__':
    main()
