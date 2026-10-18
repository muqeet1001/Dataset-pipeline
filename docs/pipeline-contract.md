# Automated pipeline contract

The permanent requirement is one program invocation that processes dataset records
one at a time, creates and verifies each mesh, and passes that record to the next
configured stage before moving to the next record. New stages must preserve this
flow. They should not require an operator to move files or invoke another command
for every subject.

## Entry point and sequence

`python -m pipeline run` processes the complete imported dataset in its source order.
`--limit N` or repeated `--subject-id` arguments support trials. The MHR model and
fitting basis are initialized once, then the runner follows this sequence per record:

1. Check whether the saved record and fitting configuration match the current run.
2. Generate a new mesh, or verify and reuse a completed one.
3. Enforce fitting acceptance and reproduce the PLY from the saved MHR inputs.
4. Persist the artifact checksums and checkpoint atomically.
5. Write a handoff and call the next stage if one is configured.
6. Save a successful delivery receipt, then advance to the next dataset record.

Each output directory permits one runner at a time through an OS file lock. The
lock is released automatically after an interruption. A run continues after a
record failure and returns a nonzero exit code if any record is rejected or fails.

## Next-stage function

The next stage is selected with `--next-stage module:function`. It is a synchronous
local Python function that accepts one dictionary and returns a JSON-serializable
receipt. For example, its signature is `def process(handoff: dict) -> dict`.
The VStitcher clothing stage can be selected with `--clothing-config` instead.
Its preparation mode preserves pending state until actual garments and runtime
access are available. See `step-2-vstitcher.md`.

Relevant handoff fields:

| Field | Meaning |
| --- | --- |
| `subject_id` | Stable source subject identifier |
| `source_commit` | Pinned source dataset commit |
| `output_root` | Absolute root of this local run |
| `files` | Artifact paths relative to `output_root` |
| `sha256` | Verified artifact checksums |
| `units` / `vertical_axis` | Centimetres / Y up |
| `label_kind` | Estimated MHR body from anthropometry |
| `idempotency_key` | Stable key for this input, mesh, configuration and consumer |

Resolve a mesh path with `Path(handoff['output_root']) / handoff['files']['body.ply']`.
The other artifact keys are `mhr-inputs.npz`, `source-record.json`, and `fit-report.json`.
A consumer should return only after its processing and durable output are complete.
Treat handoff artifacts as immutable inputs; write downstream outputs elsewhere.
Use a different module/function ID when intentionally introducing a new consumer
version that must process previously completed meshes again.

## Restart and delivery behavior

Successful recorded deliveries are skipped on resume. If a consumer fails, the
runner preserves the verified mesh and retries delivery on the next invocation.
If interruption happens after the consumer performs a side effect but before its
receipt is saved, it may receive that input again. Consumers must deduplicate using
`idempotency_key`; delivery is at least once across interruptions.

A consumer may return `{"status": "pending", ...}` when it has only prepared a
job or is awaiting an external dependency. The runner records `pending_next_stage`
instead of successful delivery and retries on the next invocation, reusing the
verified mesh. Consumers with persistent outputs may supply a
`validate_receipt(receipt)` method; invalid or missing downstream outputs trigger
redelivery without mesh regeneration.

Changes to a record, fitting settings, fitting code, model checksum or relevant
dependency versions invalidate that record's mesh checkpoint. A different consumer
can receive an existing verified mesh. A ready handoff is withdrawn before a mesh
is regenerated, so a failed replacement cannot leave a stale handoff queued.

The original fitting scope remains height and three torso-section proxies. Automation
does not establish accuracy for the remaining body measurements. The checked-in
validation runs cover a small sample, not all 6,068 generated meshes.
