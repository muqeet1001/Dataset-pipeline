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
.\.venv\Scripts\python.exe -m pipeline.avatar_clothing --fits tight medium loose --device cpu --steps 150 --substeps 4 --iterations 12 --face-budget 1800
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

## Sewn cloth revision

The earlier level-surface draft passed geometry checks but looked rigid. It is
preserved in Git history; it is not evidence of realistic garment fit. The active
adapter now uses original flat front/back tee and trouser cut pieces with paired
sewn shoulder, side, inside-leg and crotch boundaries. The tee has a scoop neck,
short sleeve cuffs and a hem covering the waistband. Chest, hip and cuff cut
sizes distinguish tight, medium and loose fits. These are prototype patterns
for the supplied male A-pose, not a general tailoring system.

The cut pieces define triangle rest metrics, area, mass and strain reference.
All bending rest angles are zero. The curved arrangement is only the starting
state. The shirt has **no pinned collar or other fixed vertices**: shoulder and
body contact support it. Only the trouser waistband edge is supported, standing
in for an elastic waistband. Both garments run together with gravity, bending,
membrane forces, body contact, self-contact and inter-garment contact.

On CPU, custom Warp Jacobi contact constraints sample vertices, edge quarters
and triangle quadrature points against the full unchanged MHR triangle mesh
inside each physics substep. Swept vertex, edge and face samples limit those
corrections against the cloth mesh so a body correction cannot simply push one
fabric layer through another. This supplements Newton body contact. The final
cloth is not shrinkwrapped or repaired after export. Fabric properties remain
an assumed cotton-like preset; a physical swatch and reference fitting are
needed for calibration.

`drape.f32` and `drape.json` contain actual recorded simulation states, topology
ranges and physical times. **Play drape** in the viewer plays these states at
quarter speed; the timeline lets you inspect any recorded frame. The final GLB
is the final solver state. Playback never animates or rescales the body.

Style3D remains experimental; VBD is the locally tested default. No EC2 GPU
benchmark or equivalence with MD/VStitcher has been established.

## Validation scope

Reports check finite vertices, settling, body penetration samples, edge/triangle
crossings against the actual body, cloth self/intersections, strain and GLB body
roundtrip. A numerical pass is labelled `numerical_pass_unvalidated_fabric`.
`needs_review` remains a failed acceptance condition and is never training-ready.
No body hiding, body rescaling or export-only correction is used to manufacture
a pass. Max source-body roundtrip error is checked against 0.0001 mm.

The previous shell pilot passed all three fits, including independent reverse
body-edge crossing checks, but failed the user's visual realism review. The sewn
revision must be evaluated separately. `reports/local-avatar-pilot.json` contains the actual
metrics and timing scope; `reports/local-avatar-resume.json` records an actual
second launcher run that reused all three scenes without changing them.

Eight local tests cover body identity/units, unknown metadata rejection, graded
flat rest sizes, manifold sewing topology, correct garment openings, zero shirt
pins, noncollapsed arrangement triangles, a triangle crossing missed by vertex-only
containment, movable/pinned contact behavior, a body correction blocked by another fabric
layer, corrupt receipt handling and output
locking. The existing 27 project tests remain a separate regression suite.

The original complete dataset runner remains working and unchanged. This local
program starts from an existing avatar; it does not regenerate the 6,068 MHR
bodies, implement female garments, or mark the full commercial dataset ready.
Future EC2 work should reuse this tested code and add those stages through the
existing handoff contract after the remaining garment-quality gates pass.

