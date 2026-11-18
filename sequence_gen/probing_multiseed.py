"""Layer probing repeated over seeds on fixed embeddings.

Runs run_classification / run_regression from linear_probing_overall.py unchanged,
except that the ProGen embeddings of every split are extracted once and cached on
disk, so repeated seeds differ only in the probe's initialisation and batch order.
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch

import linear_probing_overall as lpo

if not torch.cuda.is_available():
    # fast_transformer_linear_probing hard-codes .to('cuda'); send those to the CPU
    _to = torch.Tensor.to
    torch.Tensor.to = lambda self, *a, **k: _to(self, *('cpu' if x == 'cuda' else x for x in a), **k)

CACHE_DIR = 'probing_multiseed/embeddings'
_Extractor = lpo.EmbeddingExtractor


def cpu_extractor(checkpoint_path, vocab_size, device='cpu'):
    """EmbeddingExtractor.__init__ without the hard-coded .cuda()."""
    ex = _Extractor.__new__(_Extractor)
    base_model = lpo.CTRLmodel(vocab_size=vocab_size, embedding_dim=1280)
    lora_config = lpo.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["Wq", "Wk"], bias="none")
    ex.model = lpo.get_peft_model(base_model, lora_config)
    ex.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])
    ex.model.eval()
    ex.num_layers = ex.model.num_layers
    ex.device = 'cpu'
    return ex


class CachedExtractor:
    """Drop-in for EmbeddingExtractor; loads ProGen only on a cache miss."""
    def __init__(self, checkpoint_path, vocab_size, device='cuda'):
        self.args = (checkpoint_path, vocab_size, device)
        self.extractor = None

    def extract_all_layers_mean_pool(self, sequences, tokenizer, batch_size=32):
        key = hashlib.md5('\n'.join(sequences).encode()).hexdigest()[:16]
        path = os.path.join(CACHE_DIR, f'{key}_n{len(sequences)}.npy')
        if not os.path.exists(path):
            if self.extractor is None:
                self.extractor = _Extractor(*self.args) if torch.cuda.is_available() else cpu_extractor(*self.args)
            layers = self.extractor.extract_all_layers_mean_pool(sequences, tokenizer, batch_size)
            arr = np.stack([layers[i] for i in range(len(layers))]).astype(np.float32)
            np.save(path + '.tmp.npy', arr)
            os.replace(path + '.tmp.npy', path)
            print(f'[cache] wrote {path} {arr.shape}', flush=True)
        else:
            print(f'[cache] read {path}', flush=True)
        arr = np.load(path, mmap_mode='r')
        return {i: arr[i] for i in range(arr.shape[0])}


lpo.EmbeddingExtractor = CachedExtractor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def only_layers(train_fn, layers, seed, empty):
    """Train the probe only for the chosen layers, reseeding per layer so a result
    does not depend on which other layers ran in the same job."""
    def wrapped(*a, layer_idx=0, **k):
        if layer_idx not in layers:
            return dict(empty)
        set_seed(1000 * seed + layer_idx)
        return train_fn(*a, layer_idx=layer_idx, **k)
    return wrapped


def to_float(results):
    return {layer: {k: float(v) for k, v in r.items() if k != 'metrics_data' and v is not None}
            for layer, r in results.items() if not np.isnan(r.get('test_mcc', r.get('R2_test', 0.0)))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_csv_file', default='classification.csv')
    parser.add_argument('--reg_csv_file', default='E_coli_DBAASP_with_independent_partition4.csv')
    parser.add_argument('--checkpoint', default='finetuned_progen.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mic_scaler', default='mic_scaler.pkl')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--task', choices=['cls', 'reg'], required=True)
    parser.add_argument('--layers', default='0-35', help='e.g. 0-3; empty range = only cache embeddings')
    parser.add_argument('--output_root', default='probing_multiseed')
    args = parser.parse_args()

    a, b = map(int, args.layers.split('-'))
    layers = set(range(a, b + 1))
    nan = float('nan')
    lpo.train_and_evaluate_layer_cls = only_layers(
        lpo.train_and_evaluate_layer_cls, layers, args.seed,
        dict(best_epoch=nan, best_val_mcc=nan, best_val_acc=nan, test_mcc=nan, test_acc=nan,
             test_auc=nan, test_recall_pos=nan, test_recall_neg=nan))
    lpo.train_and_evaluate_layer_reg = only_layers(
        lpo.train_and_evaluate_layer_reg, layers, args.seed,
        dict(best_val_loss=nan, MSE_test=nan, RMSE_test=nan, R2_test=nan, PCC_test=nan))

    os.makedirs(CACHE_DIR, exist_ok=True)
    task = {'cls': 'classification', 'reg': 'regression'}[args.task]
    out = os.path.join(args.output_root, f'seed_{args.seed}')
    os.makedirs(os.path.join(out, task), exist_ok=True)
    if args.task == 'cls':
        res = lpo.run_classification(args.cls_csv_file, args.checkpoint, args.batch_size,
                                     os.path.join(out, task), args.epochs)
    else:
        res = lpo.run_regression(args.reg_csv_file, args.checkpoint, args.batch_size,
                                 os.path.join(out, task), args.epochs, args.mic_scaler)
    if layers:
        path = os.path.join(out, f'{task}_L{a}-{b}.json')
        with open(path, 'w') as f:
            json.dump({'seed': args.seed, task: to_float(res)}, f, indent=1)
        print(f'[done] seed {args.seed} {task} layers {a}-{b} -> {path}')
