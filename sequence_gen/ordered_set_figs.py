"""The P1-P18 attention figures, every residue labelled.

Two review points are addressed here.  The panels of the previous versions were
vertically stretched, and the AMP tag was drawn as part of the map, where it
reads as a zeroth residue and compresses the colour scale of the residue pairs
(see split_amp_figs.py, which does the same for the reference peptides).

    python3 ordered_set_figs.py
"""
import os
import string

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from motif_common import sym
from split_amp_figs import render

plt.rcdefaults()

CACHE = os.path.join('top10_layers', 'attn_all_layers.npz')
OUT = os.path.join('..', '69c005365ec28e7b5ef2bc4c', 'Figures')
LAYER = 22

SEQ = {
    'P1':  "KWRSFIRGVTHAAHTALKHYFRRKRSG",
    'P2':  "LWGHIFRGIKHIGKTIHRLVTG",
    'P3':  "RWRSILKHGLKHIGKTIHRLVTG",
    'P4':  "VWRHFFRGIRHVGKTIHRLVTG",
    'P5':  "VWQHIFRGIVHVGRTIHRLITG",
    'P6':  "VTWRHIFRGIVHVGKTIHRLVTG",
    'P7':  "VKSWIRVAGRAIGKLIRRK",
    'P8':  "IWHHIVRGALHVGKLIHRLVTG",
    'P9':  "FIHHIIGGLFSAGKAIHRLIRRRRR",
    'P10': "FIHHIIGGLFSAGRAIHRLIRRRRR",
    'P11': "VTWRAIFRGIIHVGKTIHRLVTG",
    'P12': "LWHIFKRIHRGIVHVGKTIHRLVTG",
    'P13': "RWKGVIQAGKAIHRIIRRRR",
    'P14': "VFWAKFKVAFRGVRHIGRTILRHVLR",
    'P15': "VWGFIKHIVRPILRRLTRGLNKL",
    'P16': "NWRNAWRRFKGAFRHVIHTAKRF",
    'P17': "DWRFIRRIWKVGSHIYRAIKKLL",
    'P18': "FIHHIIWILFSAGRAIHRLIRRRRR",
}

SETS = [('ordered_set1_split.png', ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']),
        ('ordered_set2_split.png', ['P7', 'P8', 'P9', 'P10', 'P11', 'P12']),
        ('ordered_set3_split.png', ['P13', 'P14', 'P15', 'P16', 'P17', 'P18'])]


def main():
    cache = np.load(CACHE)
    key = {k.split(' ')[0]: k for k in cache.files}
    for fname, names in SETS:
        entries = []
        for i, name in enumerate(names):
            M = sym(cache[key[name]][LAYER])   # this cache is already head-averaged
            entries.append((f'{string.ascii_uppercase[i]}   {name}',
                            SEQ[name], M, np.abs(M[1:, 1:]).max()))
        render(os.path.join(OUT, fname), entries)


if __name__ == '__main__':
    main()
