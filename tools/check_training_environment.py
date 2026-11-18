#!/usr/bin/env python3
"""Check imports, one small CPU LoRA training step, and optional checkpoint shapes."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from torch.utils.tensorboard import SummaryWriter
    from torch._subclasses.fake_tensor import FakeTensorMode

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'sequence_gen'))
    import pytorch_transformer_gpu as transformer

    class TiedEmbeddingSoftmax(torch.nn.Module):
        # The archived generation script executes main() on import, so retain
        # its two tensor definitions here without importing that entry point.
        def __init__(self, vocab_size, width):
            super().__init__()
            self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, width)))
            self.b = torch.nn.Parameter(torch.zeros(vocab_size))

        def forward(self, inputs, embed=True):
            if embed:
                return torch.nn.functional.embedding(inputs, self.w)
            return torch.tensordot(inputs, self.w.t(), 1) + self.b

    class Model(torch.nn.Module):
        def __init__(self, vocab_size, layers=1, width=32, heads=4, dff=64):
            super().__init__()
            self.tied_embedding_softmax = TiedEmbeddingSoftmax(vocab_size, width)
            self.encoder = transformer.Encoder(num_layers=layers, d_model_size=width,
                                               num_heads=heads, dff=dff, input_vocab_size=151)

        def forward(self, tokens):
            embedded = self.tied_embedding_softmax(tokens)
            return self.tied_embedding_softmax(self.encoder(embedded), embed=False)

    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                        target_modules=['Wq', 'Wk'], bias='none')
    torch.manual_seed(313)
    torch.set_num_threads(1)
    model = get_peft_model(Model(64), config)
    optimizer = torch.optim.Adam(model.parameters())
    # Keep the environment test inexpensive and usable without a GPU allocation.
    accelerator = Accelerator(cpu=True)
    model, optimizer = accelerator.prepare(model, optimizer)
    tokens = torch.randint(0, 64, (2, 8))
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64),
                                             tokens[:, 1:].reshape(-1))
    accelerator.backward(loss)
    gradients = [parameter.grad for name, parameter in model.named_parameters()
                 if 'lora_' in name and parameter.grad is not None]
    if not gradients or not any(torch.count_nonzero(gradient).item() for gradient in gradients):
        raise RuntimeError('LoRA adapters did not receive gradients')
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite smoke-test loss')
    optimizer.step()
    with tempfile.TemporaryDirectory() as directory:
        with SummaryWriter(directory) as writer:
            writer.add_scalar('smoke/loss', loss.item(), 0)

    report = dict(python=sys.version.split()[0],
                  packages={name: importlib.metadata.version(name) for name in
                            ['torch', 'numpy', 'peft', 'transformers', 'accelerate',
                             'tensorboard', 'tqdm', 'huggingface-hub', 'tokenizers', 'safetensors']},
                  cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                  cpu_lora_training_step=True, smoke_loss=loss.item(), checkpoints={})
    if args.checkpoint_dir:
        # Fake tensors validate every name and shape without allocating 14 GB of weights.
        for name in ['pretrain_progen_full.pth', 'finetuned_progen.pth', 'finetuned_progen_full.pth']:
            with FakeTensorMode():
                checkpoint = torch.load(args.checkpoint_dir / name, map_location='cpu', weights_only=True)
            state = checkpoint.get('model', checkpoint)
            with torch.device('meta'):
                expected = Model(129407, layers=36, width=1280, heads=16, dff=8192)
                if 'model' in checkpoint:
                    expected = get_peft_model(expected, config)
            expected_shapes = {key: tuple(value.shape) for key, value in expected.state_dict().items()}
            actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
            if expected_shapes != actual_shapes:
                raise RuntimeError('Checkpoint names/shapes do not match architecture: ' + name)
            report['checkpoints'][name] = dict(tensors=len(state), names_and_shapes_match=True,
                                              contains_optimizer='optimizer' in checkpoint)
    if args.require_cuda and not report['cuda_available']:
        raise RuntimeError('CUDA is unavailable; use an NVIDIA GPU allocation')
    report['limits'] = ['Small CPU model used for the training step',
                        'Checkpoint tensor names and shapes checked; full-model CUDA inference and multi-GPU training not rerun']
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    with working_directory(root / 'sequence_gen'):
        main()
