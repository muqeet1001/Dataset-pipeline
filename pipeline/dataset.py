"""Read the imported ANSUR II records with explicit units and stable IDs."""
import csv
import hashlib
import json
import math
from pathlib import Path


def load_subjects(root: Path):
    manifest = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    subjects = []
    seen = set()
    for filename, expected in manifest['files'].items():
        path = root / filename
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {filename}')
        sex = filename.removeprefix('ansur2_').removesuffix('.csv')
        count = 0
        with path.open(encoding='latin-1', newline='') as stream:
            reader = csv.DictReader(stream)
            headers = [name.lower() for name in reader.fieldnames]
            numeric_headers = headers[:headers.index('gender')]
            for source_line, raw in enumerate(reader, 2):
                row = {key.lower(): value for key, value in raw.items()}
                identifier = f'ansur2_{sex}_{row["subjectid"]}'
                if identifier in seen:
                    raise ValueError(f'Duplicate subject: {identifier}')
                seen.add(identifier)
                measurements = {
                    key: float(row[key]) / 10.0
                    for key in numeric_headers if key not in ('subjectid', 'weightkg')
                }
                mass = float(row['weightkg']) / 10.0
                if not all(math.isfinite(value) and value > 0 for value in (*measurements.values(), mass)):
                    raise ValueError(f'Invalid measurement: {identifier}')
                subjects.append({
                    'subject_id': identifier, 'sex': sex, 'measurements_cm': measurements,
                    'mass_kg': mass, 'source_file': filename, 'source_line': source_line,
                    'source_commit': manifest['source_commit'],
                })
                count += 1
        if count != expected['rows']:
            raise ValueError(f'Row count differs: {filename}')
    if len(subjects) != manifest['total_records']:
        raise ValueError('Dataset total differs')
    return subjects


def normalize(root: Path, output: Path):
    subjects = load_subjects(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='\n') as stream:
        for subject in subjects:
            stream.write(json.dumps(subject, sort_keys=True) + '\n')
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'records': len(subjects), 'sha256': digest}
