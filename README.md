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

This commit completes only the dataset import. MHR integration is the next step.
