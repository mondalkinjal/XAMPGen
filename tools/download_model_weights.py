#!/usr/bin/env python3
"""Download or verify the unchanged, single-file XAMPGen checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record['bytes']:
        raise ValueError('Checkpoint size differs from the original: ' + str(path))
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != record['sha256']:
        raise ValueError('Checkpoint SHA-256 differs from the original: ' + str(path))


def download(record, output_dir, url=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / record['name']
    if destination.exists():
        verify(destination, record)
        print('Verified existing original checkpoint: ' + str(destination))
        return
    url = url or record.get('download_url')
    if not url:
        raise ValueError('A public host for the single original checkpoint has not been configured. '
                         'See docs/MODEL_WEIGHTS.md or pass --url with its download link.')
    pending = destination.with_name(destination.name + '.download')
    if urllib.parse.urlparse(url).hostname in {'drive.google.com', 'drive.usercontent.google.com'}:
        try:
            import gdown
        except ImportError as error:
            raise ValueError('Install the Drive downloader with: '
                             'python -m pip install -r requirements-model-download.txt') from error
        downloaded = gdown.download(url=url, output=str(pending), fuzzy=True,
                                    quiet=False, use_cookies=False)
        if not downloaded:
            raise ValueError('Drive download failed. Check that the file allows '
                             'Anyone with the link to view, or download it in your browser.')
        verify(pending, record)
        pending.replace(destination)
        print('Downloaded and verified original checkpoint: ' + str(destination))
        return
    checksum = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=120) as source:
        with pending.open('wb') as target:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                target.write(block)
                checksum.update(block)
                size += len(block)
                if size > record['bytes']:
                    raise ValueError('Download exceeds original checkpoint size')
    if size != record['bytes'] or checksum.hexdigest() != record['sha256']:
        raise ValueError('Downloaded file does not match the original checkpoint')
    pending.replace(destination)
    print('Downloaded and verified original checkpoint: ' + str(destination))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=root / 'models/model_manifest.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'sequence_gen')
    parser.add_argument('--url', help='Download URL for the single unchanged checkpoint')
    parser.add_argument('--verify-only', type=Path, help='Verify an existing checkpoint without downloading')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema_version'] != 2 or len(manifest['models']) != 1:
        parser.error('Expected the single-checkpoint manifest, schema version 2')
    model = manifest['models'][0]
    if args.verify_only:
        verify(args.verify_only, model)
        print('Verified unchanged original checkpoint: ' + str(args.verify_only))
    else:
        download(model, args.output_dir, args.url)


if __name__ == '__main__':
    main()
