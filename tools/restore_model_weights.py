#!/usr/bin/env python3
"""Verify and restore byte-exact model checkpoints from the release parts."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def matches(path, record):
    return (path.is_file() and path.stat().st_size == record['bytes']
            and digest(path) == record['sha256'])


def restore(model, parts_dir, output_dir, base_url=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / model['name']
    if target.exists():
        if matches(target, model):
            print('Already verified: ' + str(target))
            return
        raise ValueError('Existing checkpoint does not match manifest: ' + str(target))
    verified_parts = []
    for part in model['parts']:
        path = parts_dir / part['name']
        if not matches(path, part):
            url = base_url or model.get('download_base_url')
            if not url:
                raise ValueError('Missing or invalid part %s. Supply --parts-dir or --base-url; '
                                 'public weight hosting has not been configured.' % part['name'])
            parts_dir.mkdir(parents=True, exist_ok=True)
            pending = path.with_name(path.name + '.download')
            with urllib.request.urlopen(url.rstrip('/') + '/' + part['name'], timeout=120) as source:
                with pending.open('wb') as handle:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        handle.write(block)
            if not matches(pending, part):
                raise ValueError('Downloaded part failed verification: ' + part['name'])
            pending.replace(path)
        verified_parts.append(path)
    pending = target.with_name(target.name + '.assembling')
    with pending.open('wb') as handle:
        for path in verified_parts:
            with path.open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    handle.write(block)
    if not matches(pending, model):
        raise ValueError('Assembled checkpoint failed verification: ' + model['name'])
    pending.replace(target)
    print('Verified and restored: ' + str(target))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root / 'models/model_manifest.json')
    parser.add_argument('--parts-dir', type=Path, default=root / 'model-parts')
    parser.add_argument('--output-dir', type=Path, default=root / 'sequence_gen')
    parser.add_argument('--model', action='append', help='Checkpoint filename; repeat to select several')
    parser.add_argument('--base-url', help='Public release URL containing the named parts')
    args = parser.parse_args()
    models = json.loads(args.manifest.read_text())['models']
    known = {model['name'] for model in models}
    if args.model and set(args.model) - known:
        parser.error('Unknown model: ' + ', '.join(sorted(set(args.model) - known)))
    for model in models:
        if not args.model or model['name'] in args.model:
            restore(model, args.parts_dir, args.output_dir, args.base_url)


if __name__ == '__main__':
    main()
