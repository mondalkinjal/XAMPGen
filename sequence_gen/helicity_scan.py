"""Layer-22 attention, heads averaged, for the 18 peptides -> helicity_cache.npz"""
import numpy as np
import torch

import motif_head_lib as L

LAYER = 22
PEPTIDES = [
    ("P1", "KWRSFIRGVTHAAHTALKHYFRRKRSG", 1.000),
    ("P2", "LWGHIFRGIKHIGKTIHRLVTG", 0.778),
    ("P3", "RWRSILKHGLKHIGKTIHRLVTG", 0.825),
    ("P4", "VWRHFFRGIRHVGKTIHRLVTG", 0.737),
    ("P5", "VWQHIFRGIVHVGRTIHRLITG", 0.820),
    ("P6", "VTWRHIFRGIVHVGKTIHRLVTG", 0.639),
    ("P7", "VKSWIRVAGRAIGKLIRRK", 0.405),
    ("P8", "IWHHIVRGALHVGKLIHRLVTG", 0.689),
    ("P9 (WT)", "FIHHIIGGLFSAGKAIHRLIRRRRR", 0.521),
    ("P10", "FIHHIIGGLFSAGRAIHRLIRRRRR", 0.599),
    ("P11", "VTWRAIFRGIIHVGKTIHRLVTG", 0.740),
    ("P12", "LWHIFKRIHRGIVHVGKTIHRLVTG", 0.760),
    ("P13", "RWKGVIQAGKAIHRIIRRRR", 0.411),
    ("P14", "VFWAKFKVAFRGVRHIGRTILRHVLR", 0.632),
    ("P15", "VWGFIKHIVRPILRRLTRGLNKL", 0.500),
    ("P16", "NWRNAWRRFKGAFRHVIHTAKRF", 0.440),
    ("P17", "DWRFIRRIWKVGSHIYRAIKKLL", 0.551),
    ("P18", "FIHHIIWILFSAGRAIHRLIRRRRR", 0.667),
]


def main():
    model, dev = L.load_model()
    cache = {}
    for name, seq, _ in PEPTIDES:
        _, attn = L.run(model, torch.tensor([L.encode(seq)], device=dev))
        cache[name] = attn[LAYER][0].float().mean(0).cpu().numpy()
    np.savez_compressed('motif_head_results/helicity_cache.npz', **cache)


if __name__ == '__main__':
    main()
