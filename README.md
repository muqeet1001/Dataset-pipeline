# Dataset-pipeline

## Step 1: Import Mahmoud dataset

The original ANSUR II measurement dataset has been imported from
`Twinnie-Inc/TWINNIE`, branch `Mahmoud-Dataset-5.8`, commit
`2f901e81447103dba946b301d844f0fd6476a710`.

Source folder: `Scripts/Dataset/fixtures/ansur2`.
Local folder: `data/raw/ansur2`.

- `ansur2_male.csv`: 4,082 records, 108 columns.
- `ansur2_female.csv`: 1,986 records, 108 columns.
- Total: 6,068 records.

Both CSV files and the original `SOURCE.md` are preserved byte-for-byte.
`provenance.json` records the exact source and verified SHA-256 checksums.
The original `SOURCE.md` checksums use CRLF line endings; the imported Git blobs
use LF. Content equivalence was verified, and both hashes are recorded.

Verify this import with Python 3:

```sh
python scripts/verify_dataset.py
```

The CSVs use Latin-1 encoding. Linear measurements are in millimetres;
`weightkg` is in tenths of a kilogram. The original source documentation
explains the dataset provenance and Army population scope.

## Step 2: Official MHR source

The full official Meta MHR source is in `third_party/MHR`, pinned at commit
`d96fafa33bbf018647c70c3525e91f53e79d2a14`. Its license and original files are
preserved. See `THIRD_PARTY.md`.

## Step 3: Test MHR mesh generation

From this repository root, create a Python 3.11+ environment and install dependencies:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_mhr_assets.py
python -m pipeline smoke-mhr
```

The downloader verifies the official `v1.0.1` archive checksum, then extracts its
TorchScript model. This supported MHR backend runs on CPU and embeds the mesh
topology, so it needs no separate PyMomentum installation.

The smoke test verifies finite vertices and skeleton, deterministic inference,
response to an identity parameter, valid triangle indices, and PLY export/reload.
`examples/smoke/neutral.ply` and `examples/smoke/smoke-report.json` contain the
verified result: 18,439 vertices and 36,874 triangles. Mesh units are centimetres,
with Y as the vertical axis.

Downloaded model assets and ordinary runs stay outside Git. Each completed step
is committed separately.
