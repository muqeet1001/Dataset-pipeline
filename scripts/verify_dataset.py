"""Verify the imported source bytes and CSV dimensions using only Python stdlib."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'ansur2'
manifest = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
total = 0
for name, expected in manifest['files'].items():
    content = (root / name).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected['sha256']:
        raise ValueError(f'Checksum mismatch: {name}')
    with (root / name).open(encoding='latin-1', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f'Invalid row width: {name}, record {count + 1}')
            count += 1
    if len(header) != expected['columns'] or count != expected['rows']:
        raise ValueError(f'CSV dimensions differ: {name}')
    total += count
    print(f'{name}: {count:,} records, {len(header)} columns, SHA-256 verified')
if total != manifest['total_records']:
    raise ValueError('Total record count differs')
print(f'PASS: {total:,} total records; source {manifest["source_commit"]}')
