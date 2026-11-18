"""Model plumbing for the motif figure pipeline.

Loads the finetuned ProGen/CTRL checkpoint and runs single sequences through it,
returning the per-layer attention.  The model modules in
fast_transformer_extraction.py are used exactly as defined -- nothing is patched.

    load_model()       -> (model, DEVICE), LoRA-wrapped, eval mode
    encode(seq)        -> token id list  [AMP_kw] + residues
    run(model, tokens) -> logits, all_attn   (all_attn[layer] = [B, 16, L, L])
"""
from __future__ import print_function

import os
import pickle
import torch

import fast_transformer_extraction as fte
from peft import LoraConfig, get_peft_model

VOCAB_PATH = 'mapping_files/vocab.txt'
CKPT = 'finetuned_progen.pth'
EMBED_DIM = 1280
NUM_LAYERS = 36
NUM_HEADS = 16

_vocab = open(VOCAB_PATH, encoding='utf-8').read().split('\n')[:-1]
VOCAB_SIZE = len(list(map(lambda x: x.split(' ')[0], _vocab)))

with open('mapping_files/aa_to_ctrl_idx.p2', 'rb') as fh:
    AA_TO_IDX = pickle.load(fh)
with open('mapping_files/kw_to_ctrl_idx.p2', 'rb') as fh:
    KW_TO_IDX = pickle.load(fh)
with open('mapping_files/kw_to_name.p2', 'rb') as fh:
    KW_TO_NAME = pickle.load(fh)

_amp_key = [k for k, v in KW_TO_NAME.items() if v == 'Antimicrobial'][0]
AMP_CTRL_ID = KW_TO_IDX[_amp_key]

# ---------------------------------------------------------------------- model

class TiedEmbeddingSoftmax(torch.nn.Module):
    def __init__(self, vocab_size=None, embedding_size=EMBED_DIM):
        super().__init__()
        vocab_size = VOCAB_SIZE if vocab_size is None else vocab_size
        self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
        self.b = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs, embed=True):
        if embed:
            return torch.nn.functional.embedding(inputs, self.w)
        return torch.tensordot(inputs, self.w.t(), 1) + self.b


class CTRLmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tied_embedding_softmax = TiedEmbeddingSoftmax()
        self.encoder = fte.Encoder()

    def forward(self, inputs, past=None):
        x = self.tied_embedding_softmax(inputs, embed=True)
        x, new_past, all_attn = self.encoder(x, past)
        return self.tied_embedding_softmax(x, embed=False), new_past, all_attn


def load_model(ckpt=CKPT, verbose=True):
    import time
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f'[load {time.time() - t0:6.1f}s] {msg}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'device={device} threads={torch.get_num_threads()}')
    model = CTRLmodel()
    log('model built')
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["Wq", "Wk"], bias="none")
    model = get_peft_model(model, cfg)
    log('LoRA wrapped')
    state = torch.load(ckpt, map_location=device)
    log(f'checkpoint read ({os.path.getsize(ckpt) / 1e9:.1f} GB)')
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(device).eval()
    log('weights in place')

    if verbose:
        print(f'[model] {ckpt} on {device} | {NUM_LAYERS} layers x {NUM_HEADS} heads', flush=True)
    return model, device


# ------------------------------------------------------------------ sequences

def encode(seq):
    """[AMP keyword] + residues.  Token 0 is the control code, residue i sits at
    token index i+1."""
    return [AMP_CTRL_ID] + [AA_TO_IDX[a] for a in seq.upper() if a in AA_TO_IDX]


@torch.no_grad()
def run(model, tokens):
    logits, _, all_attn = model(tokens, past=None)
    return logits, all_attn
