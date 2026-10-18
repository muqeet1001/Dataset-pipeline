# Cloth technology decision — 18 October 2026

Recommend **modern Newton/Warp GPU simulation on EC2, with Three.js viewing**.
Benchmark Newton's **SolverStyle3D first and SolverVBD alongside it** before
selecting the production solver. This is an engineering recommendation, not a
measured speed or garment-quality winner. The user confirmed future commercial
model training and accepted Python/GPU simulation outside the browser.

The immediate goal is a fixed MHR body wearing a shirt and pants in three fit
conditions. Motion/video generation remains a later stage. An engine predicts
cloth deformation; garment construction, grading and automatic placement are
separate work. None of the reviewed projects is a complete MHR dressing pipeline.

## Comparison for this task

| Technology | Relevant evidence | Decision |
|---|---|---|
| Newton **SolverStyle3D** | Official example loads an actual garment, its panel coordinates and an avatar collider. Solver source is projective dynamics with directional stretch/bending. | First commercial implementation candidate. Need MHR placement, pants, two-garment validation and timed export. |
| Newton **SolverVBD** | Source exposes self-contact; upstream tests exercise vertex–triangle and edge–edge collision detection. | Compare with Style3D on identical outfits; generic cloth support does not establish correct garment fit. |
| **ContourCraft** | Learned multi-garment method with explicit intersection handling; arbitrary body mesh sequence interface exists. | Stronger learned candidate than plain HOOD for overlapping garments, but the supplied collision dependency has research-use restrictions. Exclude from the commercial default. |
| **HOOD** | Generalizes garment geometry/materials, but original method does not model garment–garment interactions/self-collisions. | Useful research baseline; insufficient for the required shirt/pants interaction alone. |
| **GarmentCode + its Warp fork** | Parametric sewing patterns and automated static draping are close to this workflow. | Pattern generation is useful to assess separately. The supplied old simulator is restricted to research/evaluation; full outfits also require extensions. |
| **MD / VStitcher / CLO** | Garment-focused reference workflow; current MD documentation includes CPU/GPU modes and accurate-fabric fitting. | Preserve as optional quality references. The reported 40 minutes includes setup and export; it is not a physics-only benchmark. User has put these products aside as the active backend. |
| **Blender cloth** | Documented collider and self-collision workflow. | Useful independent physical reference; still needs automated patterns/placement. No measured throughput advantage here. |
| **Three.js/WebGPU or Ammo cloth examples** | Official WebGPU example describes a simple Verlet cloth system. | Viewer choice remains Three.js. Building a complete browser garment solver adds unnecessary scope now that EC2 simulation is allowed. |
| **SNUG** | Released inference uses trained garment models and SMPL. | More adaptation for arbitrary MHR bodies and new garment styles; not the default. |
| **NeuralGarSim** | 2026 project demonstrates quasi-static deformation across representations. | Watchlist. Reviewed page's code link was unavailable and paper link returned the project page; no verified MHR body/contact/outfit implementation from that evidence. |

These decisions follow the primary sources linked below. They are not an
exhaustive claim about every cloth simulator or a head-to-head runtime test.

## Published tests and their practical limits

ContourCraft evaluates 40 sequences across five unseen, layered outfits. Its
Table 2 reports 3.66–10.49 inference frames/s on an RTX 4090 for outfits containing
12,030–36,515 garment nodes. The perceptual study scored CLO3D 4.29 and ContourCraft
4.15 out of five; these are ratings, not percentage accuracy or evidence of equal
physical fit. The paper identifies failures with ambiguous layer ordering and
small non-manifold garment parts. Its results do not certify MHR or ANSUR extremes.
[ContourCraft paper, sections 5–7 and Table 2](https://arxiv.org/pdf/2405.09522).

GarmentCodeData reports approximately 30 seconds of simulation per sample on an
RTX 3090, about 72% overall simulation success including mesh/quality failures,
and a five-minute simulation cap. Its limitations include full outfits, layers,
hardware details and fabric calibration. This supports its relevance as a static
draping reference, not a promise of a 30-second complete outfit on our EC2 host.
[GarmentCodeData paper, sections 3.4 and 5](https://cg.cs.tu-dortmund.de/publications/2024-garment.pdf).

Newton's inspected Style3D tests cover matrix precomputation, topology rejection,
finite zero-mass handling, particle flags and gravity. Its garment example's
final check is a bounding volume check. The broader cloth collision suite checks
contact detection and filtering. These are useful component tests; they do not
test anatomical placement, three fit sizes or our dataset. **Upstream tests were
read, not executed in this review.**
[Style3D tests](https://github.com/newton-physics/newton/blob/f49482ce07e20764110fd147a67c7334eeae638d/newton/tests/test_solver_style3d.py),
[cloth collision tests](https://github.com/newton-physics/newton/blob/f49482ce07e20764110fd147a67c7334eeae638d/newton/tests/test_collision_cloth.py),
[garment example](https://github.com/newton-physics/newton/blob/f49482ce07e20764110fd147a67c7334eeae638d/newton/examples/cloth/example_cloth_style3d.py).

## Commercial dependency findings

ContourCraft's main code is MIT, but CCCollisions explicitly states that the
torch-mesh-isect license also applies. That license restricts commercial use
and commercial neural-network training. A main-repository MIT label does not
resolve this dependency. Do not run the stock stack for this commercial dataset
without suitable rights or a properly independent replacement.
[CCCollisions notice](https://github.com/Dolorousrtur/CCCollisions/tree/215195a7716810c2a6da069724409e5286a250d9),
[referenced license](https://github.com/vchoutas/torch-mesh-isect/blob/85b30177821a1527e3fe62fcf8ce65262d7c1879/LICENSE).

GarmentCode's main repository is MIT. Its required NvidiaWarp-GarmentCode fork
is based on Warp 1.0.0-beta.6, and license section 3.3 limits use to research or
evaluation. Do not confuse that fork with current Warp/Newton. Pattern code,
datasets, body assets and simulator each need their own recorded provenance.
[GarmentCode license](https://github.com/maria-korosteleva/GarmentCode/blob/d449629979028123a5c4dc9e732a2ec19b7fce31/LICENSE),
[fork license](https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode/blob/63baf6855efdd89b2834b74640f84b3bb0d86b50/LICENSE.md).

Newton's repository and inspected solver/example files specify Apache-2.0.
Use appropriately licensed garments and inspect the actual pinned dependencies
and downloaded example assets before incorporating them. This review did not
establish rights to every possible garment/checkpoint asset.
[Newton license and third-party notices](https://github.com/newton-physics/newton/tree/f49482ce07e20764110fd147a67c7334eeae638d).

## Integration issues found in source

- **ContourCraft can read non-SMPL geometry:** its mesh loader uses `verts`
  `[frames, vertices, 3]` and `faces`; the notebook requires an already aligned
  garment. It does not automatically construct/grade a shirt and pants. Its bare
  mesh path still imports SMPL-related modules. The example YAML contains stale
  body-mode fields; use the actual typed config/helper, not a copied YAML.
  The environment file includes Linux-specific packages and CCCollisions needs a
  CUDA compiler. Native Windows EC2 compatibility has not been established.
  [mesh loader](https://github.com/Dolorousrtur/ContourCraft/blob/d606432c2f7e96b05664e6a4ffb9d1cb98705baf/datasets/from_any_pose.py),
  [notebook](https://github.com/Dolorousrtur/ContourCraft/blob/d606432c2f7e96b05664e6a4ffb9d1cb98705baf/Inference_from_mesh_sequence.ipynb),
  [installation](https://github.com/Dolorousrtur/ContourCraft/blob/d606432c2f7e96b05664e6a4ffb9d1cb98705baf/INSTALL.md).
- **GarmentCode needs an MHR adapter:** its simulation code multiplies body
  coordinates by 100 and selects a built-in body segmentation. Passing our
  centimetre OBJ directly would introduce an incorrect scale. MHR needs its own
  landmarks/segmentation and garment measurements. Built-in SMPL/CAESAR indices
  cannot be reused on MHR.
  [garment loader](https://github.com/maria-korosteleva/GarmentCode/blob/d449629979028123a5c4dc9e732a2ec19b7fce31/pygarment/meshgen/garment.py),
  [path/segmentation selection](https://github.com/maria-korosteleva/GarmentCode/blob/d449629979028123a5c4dc9e732a2ec19b7fce31/pygarment/meshgen/sim_config.py).
- **Newton requires actual hardware preflight:** current README supports Windows
  and Linux; fresh installs select Warp 1.18 requiring Turing+ and driver R580+.
  It documents Warp 1.17 for some older configurations. Record the EC2 GPU/driver
  and compatible versions rather than assuming its RDP session exposes CUDA.
  [pinned requirements](https://github.com/newton-physics/newton/blob/f49482ce07e20764110fd147a67c7334eeae638d/README.md).

## Benchmark using this repository's actual data

Run `python scripts/prepare_cloth_benchmark.py` to reproduce
`reports/cloth-benchmark-plan.json`. This command **only prepares cases**.
It verifies source hashes through the existing importer and selects 14 unique
subjects: the two accepted examples, plus six additional subjects per sex near
P5/P95 of height, waist/height and chest/hips. It does not generate those extra
bodies or execute cloth physics. These quantiles provide coverage, not statistical
proof for all 6,068 subjects. Add the actual range endpoints before full release.

| Stage | Subjects and conditions | Cases per backend |
|---|---|---:|
| Pilot | Existing male and female, one outfit, three fits | 6 |
| Body coverage | Twelve additional subjects, same outfit, three fits | 36 |
| Material sensitivity | Existing subjects, two additional bending presets, three fits | 12 |
| Style transfer | Existing subjects, one additional outfit, three fits | 6 |
| Total unique cases | Both garments are present in every case | 60 |

Start with the six-case pilot. Expand only after placement and quality pass.
For each pilot case measure three warm repetitions and a separately identified
cold launch. Keep the same rest geometry, material intent, body and collision
thickness across solvers; record unavoidable model differences. Benchmark both
solvers at a common mesh resolution, then repeat shortlisted cases at a finer
resolution. The JSON's profile names are unfilled test slots, not existing assets
or calibrated fabrics. No cases currently have a pass result.

| Test | Required evidence / acceptance |
|---|---|
| Commercial suitability | Pinned code/dependencies and garment assets permit intended use; exclude restricted stacks from commercial runs. |
| Body identity and units | Preserve source body hash. Compare world-space exported vertices after inverse transforms; proposed numerical tolerance 0.01 mm. Verify stature, axis and ground position. |
| Placement | Neck/arm openings on correct body parts; waistband supported; left/right legs correct. Front/back/side views plus landmark measurements. |
| True sizing | Same body/style/fabric across fits. Save graded rest dimensions/ease at chest, waist, hip, arm and thigh, plus garment lengths. Fit labels describe intent until calibrated. |
| Body contact | Independent triangle-level crossing and containment checks against full MHR; report penetration depth/area. Zero unexplained crossings; numerical contact tolerance documented separately. |
| Self / inter-garment contact | Check nonadjacent triangle pairs and shirt/pants overlap, with documented seam-adjacency exclusions. No visually inverted waistband layers or unresolved crossings. |
| Fabric behavior | Save stretch/shear/bend parameters, rest coordinates and strain distribution. Limits depend on the chosen fabric; no universal stiffness or strain value proves realism. |
| Settling / stability | Bound time and steps; save displacement/velocity over a trailing window. Proposed starting gate: 95th-percentile movement below 0.5 mm per sample over 30 samples spaced 1/60 s apart, with no growing outliers. Calibrate and lock before timing. |
| Initialization stress | Small overlap resolves or cleanly fails; wrong leg/layer ordering and impossible sizes fail rather than producing accepted examples. |
| Resolution sensitivity | Compare silhouette, clearance and strain at common and finer cloth resolution; do not count faster low-detail cloth as equivalent without checking quality. |
| Export and viewing | Reload GLB in Three.js; verify body and both named garments, units, UVs/materials, measured dimensions and hashes. Screenshots must show actual outputs. |
| Resume and failure | Repeat run, delete/corrupt one export, kill worker, change one asset/config and fail downstream delivery. Reuse valid MHR body; retry only invalid clothing; never deliver a failed variant. |
| Distribution coverage | Report pass/failure rates by sex, fit and body group, including rejected cases and reasons. Do not silently discard difficult subjects. |
| Physical-fit credibility | Compare a small licensed reference set or physical drape observations. Good appearance and zero intersections alone do not prove equivalent material accuracy to MD/VStitcher. |

All thresholds above are proposed starting engineering gates, not validated
fabric standards. Store tolerances with each run and freeze them before choosing
a winner; never relax them only to improve throughput.

Use generated-body geometry for dressing measurements. Step 1 fits stature and
chest/waist/hip perimeter proxies, not every original ANSUR measurement; source
measurements and actual MHR-surface measurements must remain distinguishable.

## Fair timing and selection

Record environment setup separately. For each cold and warm run record MHR
generation/reuse, garment generation/grading, placement, solver initialization,
simulation, validation, export and viewing load separately, plus total elapsed
time. Include retries and failures. Report median, p95, peak VRAM and successful
outfits per GPU-hour; record GPU model, driver, software SHA and mesh counts.
Compare a matched complete outfit/fit against the user's 2,400-second end-to-end
baseline. Report all three variants per subject separately and together.
Automating placement/export could reduce the vendor workflow's time too; the
40-minute observation alone does not establish that its solver is the bottleneck.

Choose only among candidates that pass the required quality gates on the same
cases. Then prefer lower total time per accepted outfit and lower manual work.
If Style3D wins on garments, use it; if VBD proves more robust at acceptable speed,
use VBD. If neither reaches sufficient quality, report that result and keep the
vendor/reference route available. Do not promise a particular speedup or
MD/VStitcher-equivalent results before this benchmark.

## Evidence and remaining work

`reports/cloth-source-review.json` records inspected source hashes and commit
pins. No third-party solver was imported or run in this review. No EC2 CUDA
benchmark, full garment pilot, trained checkpoint download or 6,068-body batch
was performed. Existing pipeline tests verify their own scope, not cloth realism.
Local verification: 27 existing project tests passed; regenerated case plans were
byte-identical, with seven subjects per sex, 20 cases per fit and all 60 statuses
remaining `not_run`. See `reports/cloth-review-validation.json`.

The current implementation instruction is
`docs/ec2-gpu-clothing-handoff.txt`. The earlier browser-only handoff is superseded.

Other primary references:

- [HOOD project](https://dolorousrtur.github.io/hood/) and
  [paper limitations](https://arxiv.org/pdf/2212.07242).
- [GarmentCodeData project](https://igl.ethz.ch/projects/GarmentCodeData/) and
  [installation requiring the fork](https://github.com/maria-korosteleva/GarmentCode/blob/d449629979028123a5c4dc9e732a2ec19b7fce31/docs/Installation.md).
- [Three.js WebGPU cloth example](https://threejs.org/examples/webgpu_compute_cloth.html).
- [MD simulation documentation](https://support.marvelousdesigner.com/hc/en-us/articles/47358232630937-Simulation),
  [VStitcher CLI guide](https://help.browzwear.com/en/articles/13265438-vstitcher-cli-local-installation-usage-guide).
- [Blender cloth collisions](https://docs.blender.org/manual/en/5.4/physics/cloth/settings/collisions.html),
  [SNUG implementation](https://github.com/isantesteban/snug),
  [NeuralGarSim project](https://4dqv.mpi-inf.mpg.de/NeuralGarSim/).
