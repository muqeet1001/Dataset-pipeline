# Local clothing pilot on the supplied MHR male avatar

Project folder: `B:\Dataset-pipeline`. The original files from
`Downloads\MHR-male-10027` are copied into `inputs/MHR-male-10027` with provenance
and verified hashes. The body remains 18,439 vertices, 36,874 triangles and
177.6 cm high. Simulation/export use metres; the source OBJ remains unchanged.

Double-click **RUN-CLOTHING.cmd** to prepare dependencies if needed, run the
outfit's three fits, and open the viewer. Double-click **VIEW-RESULTS.cmd** to
view saved outputs without another simulation. Node/npm and Python 3.12 must be
installed for automatic setup. The prepared local environment already contains
the required dependencies.

From the project directory:

```powershell
python run_clothing.py
python run_clothing.py --view-only
```

The viewer is at <http://127.0.0.1:9086/web/clothing-viewer/>. It switches the
three fits, compares rest and simulated surfaces, offers wireframe/body-opacity
inspection, and links actual GLB exports and reports. Opacity only changes the
viewer; it does not change or omit the exported body.

The [three-fit comparison](http://127.0.0.1:9086/web/clothing-viewer/compare.html)
shows all three real exports together. Verified pilot files are committed under
`examples/clothing-pilot/MHR-male-10027`; a fresh checkout seeds the viewer's run
folder from those examples if no run folder exists. Normal runs still verify
signatures/checksums before reusing them.

For explicit simulation control:

```powershell
.\.venv\Scripts\python.exe -m pipeline.avatar_clothing --fits tight medium loose --device cpu --steps 150 --substeps 4 --iterations 8 --face-budget 1200
```

`--device cuda:0` selects NVIDIA CUDA on a compatible host. This computer has
Intel Iris Xe and no NVIDIA driver, so **only CPU execution is verified here**.
Do not present these local timings as EC2 GPU benchmarks.

Outputs are under `runs/local-avatar/MHR-male-10027/`, with an atomic overall
`manifest.json` and per-fit `scene.glb`, garment GLBs/OBJs, rest geometry NPZs,
`rest.glb` and `fit-report.json`. Source/code/config signatures and artifact
hashes control reuse. Only numerically passing, intact outputs are reused;
missing/corrupt artifacts and failed cases rerun. An output lock prevents two
current programs from writing the same job. Exit code 2 means outputs require
review, not an accepted dataset sample. Exceptions are recorded and withheld.

## What the prototype actually does

It constructs continuous outer level surfaces around a voxel draft of the supplied
body, then cuts shirt/pants openings and grades the rest geometry for three sizes.
The 6 mm draft grid is only used for constructing cloth; the simulation collider
and final validation use the original full MHR body. The tight shirt includes
6 mm of additional hem room for the underlying pants. Openings are checked by regression
tests. Physical triangle rest coordinates are stored separately from appearance
UVs. These are body-derived shell drafts, **not production tailor sewing patterns**.
There is one male outfit style; random style diversity and female adaptation have
not been implemented in this local prototype.

Both garments share one Newton VBD simulation, with gravity, membrane/bending
behavior, damping, body contact, self-contact and inter-garment contact. A narrow
collar support band and a supported waistband stabilize this static
pilot. These supports are saved in the reports/NPZs; the rest of the cloth is free.
The material preset and supports are assumptions requiring calibration.

On CPU, custom Warp Jacobi contact constraints sample vertices, edge quarters
and triangle quadrature points against the full, unchanged MHR triangle mesh
inside each physics substep. This supplements Newton's vertex/body contact;
it is not post-export shrinkwrap. Newton's alternative full-surface rigid SDF
contact path requires CUDA. The local implementation is therefore a **VBD plus
sampled mesh-contact prototype**, not a claim about unmodified Newton performance.

Style3D's six upstream component tests passed on CPU, but the first normal-offset
garment adapter produced NaNs at the first simulated frame. The revised level-surface
adapter has not been validated with Style3D. Style3D is retained as an
experimental CLI option, and VBD is the tested default. This does not establish
that Style3D itself cannot simulate MHR; it records the adapter's observed failure.

## Validation scope

Reports check finite vertices, settling, body penetration samples, edge/triangle
crossings against the actual body, cloth self/intersections, strain and GLB body
roundtrip. A numerical pass is labelled `numerical_pass_unvalidated_fabric`.
`needs_review` remains a failed acceptance condition and is never training-ready.
No body hiding, body rescaling or export-only correction is used to manufacture
a pass. Max source-body roundtrip error is checked against 0.0001 mm.

The final local pilot passed all three fits, including independent reverse
body-edge crossing checks. `reports/local-avatar-pilot.json` contains the actual
metrics and timing scope; `reports/local-avatar-resume.json` records an actual
second launcher run that reused all three scenes without changing them.

Seven local tests cover body identity/units, unknown metadata rejection, graded
rest sizes and correct garment openings, a triangle crossing missed by vertex-only
containment, movable/pinned contact behavior, corrupt receipt handling and output
locking. The existing 27 project tests remain a separate regression suite.

The original complete dataset runner remains working and unchanged. This local
program starts from an existing avatar; it does not regenerate the 6,068 MHR
bodies, implement female garments, or mark the full commercial dataset ready.
Future EC2 work should reuse this tested code and add those stages through the
existing handoff contract after the remaining garment-quality gates pass.
