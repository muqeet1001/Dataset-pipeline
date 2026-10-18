# Dataset-pipeline

## Automated run

After the one-time environment and MHR asset setup below, run:

```sh
python -m pipeline run
```

The program reads the dataset in order. For each record it generates a mesh,
checks the fit, reproduces the exported mesh from its saved MHR inputs, and writes
the verified result as a handoff for the next stage. It finishes that record before
starting the next. The model is loaded once for the run.

Run the same command again to resume. Verified completed meshes are checked and
reused. Missing or modified artifacts are regenerated, and failed records are
retried. An interrupted run keeps its completed record checkpoints.

For a small trial:

```sh
python -m pipeline run --limit 2
```

Outputs are under `runs/pipeline/subjects/<subject_id>/`. Each record includes the
mesh, exact MHR inputs, source measurements, fit report, `checkpoint.json`, and
`handoff.json`. `progress.json`, `events.jsonl`, and `run-report.json` track the run.
Rejected or failed meshes are recorded and withheld from the next stage; processing
continues with the next record and the command exits with code 2 when any record fails.

The next pipeline stage has not been defined yet. Currently, validated meshes wait
in their handoff files with status `ready_for_next_stage`. When that stage is built,
connect its Python function and every accepted mesh will be passed to it immediately:

```sh
python -m pipeline run --next-stage my_next_stage:process
```

The consumer receives a handoff dictionary with the subject ID, artifact paths,
checksums, units, and a stable idempotency key. See `docs/pipeline-contract.md`.

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

## Step 4: Dataset record to MHR mesh

```sh
python -m pipeline normalize
python -m pipeline fit --subject-id ansur2_male_10027 --subject-id ansur2_female_10037
python -m pipeline verify --directory runs/subjects
```

The importer converts all 6,068 records to centimetres and kilograms. The fitter
optimizes the first 20 MHR body identity coefficients using chest, waist and hip
section perimeters at the corresponding ANSUR height levels. It applies a uniform
scale to match standing height. Pose, head, hands and facial expression retain their
neutral defaults. All original anthropometry is retained in each source record.

The generated mesh is produced by a final call to the official MHR model. Each
subject folder contains `body.ply`, `mhr-inputs.npz`, `source-record.json`, and
`fit-report.json`. The input file records the exact coefficients, pose, expression
and scale/grounding transform needed to reproduce the exported mesh. The native
skeleton is also included and is explicitly labelled as unscaled.

For a full batch, use `python -m pipeline fit --all`. This is an explicit action;
the checked-in examples cover two records, not a completed 6,068-mesh batch.
The command exits with code 2 if any completed fit fails its 2 cm perimeter
tolerance or solver/closed-mesh checks. Failed fits retain a `rejected_fit` report.

These are **estimated bodies** fitted to height and three section-perimeter proxies.
The section definitions still need anatomical validation; matching these values
does not establish that all body dimensions match the measured person. Remaining
measurements and mass are recorded as unfitted. This stage provides an executable
measurement-to-mesh baseline for the requested initial MHR test.

The tested male and female meshes and reports are in `examples/fitted`. Preview a
generated mesh by importing `body.ply` in Blender or MeshLab, or build an interactive
preview with embedded geometry:

```sh
python scripts/preview_mesh.py --mesh runs/subjects/ansur2_male_10027/body.ply --output mesh-preview.html
```

The preview uses Three.js from a public CDN. Its geometry is embedded in the file.

Run the pipeline's tests with:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests cover source counts and units, analytic section perimeters and derivatives,
separation of disjoint limb sections, missing-section rejection, and exact stature
scaling. Official model inference and exported-mesh reproduction are checked by
the `smoke-mhr` and `verify` commands. The vendored Python/PyMomentum test suite
requires its own optional upstream environment.
