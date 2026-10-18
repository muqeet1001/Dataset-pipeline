"""Download the pinned official MHR archive and verify before extracting."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official archive')
    parser.add_argument('--output', type=Path, default=Path('assets'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / 'third_party/mhr-provenance.json').read_text(encoding='utf-8'))
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = args.archive
    if archive is None:
        archive = destination / 'mhr-v1.0.1.zip'
        temporary = archive.with_suffix('.download')
        request = urllib.request.Request(provenance['asset_url'], headers={'User-Agent': 'Dataset-pipeline'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != provenance['asset_sha256']:
        raise ValueError(f'MHR archive checksum mismatch: {digest}')
    with zipfile.ZipFile(archive) as bundle:
        # The TorchScript release contains its own topology and requires only this model.
        member = 'assets/mhr_model.pt'
        with bundle.open(member) as source, (destination / 'mhr_model.pt').open('wb') as target:
            shutil.copyfileobj(source, target)
        with bundle.open('assets/LICENSE.txt') as source, (destination / 'MHR-ASSET-LICENSE.txt').open('wb') as target:
            shutil.copyfileobj(source, target)
    model = destination / 'mhr_model.pt'
    with model.open('rb') as stream:
        model_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {
        'release': provenance['asset_release'], 'archive_sha256': digest,
        'model_sha256': model_digest,
        'model_bytes': model.stat().st_size,
    }
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
