# Step 2: VStitcher clothing integration

**Historical integration:** the user has put VStitcher aside for now. Continue
with the [active Three.js EC2 handoff](ec2-threejs-clothing-handoff.txt), which
requires the clothing simulation to run inside a browser. The native results
and prepared implementation below remain a record of earlier work.

Status: **integration prepared; native avatar import confirmed; clothing simulation pending**.
The initial preparation checkpoint is followed by a successful desktop import pilot
on 2026-10-18. The EC2 VStitcher desktop was already signed in; no password was
entered or stored by this pipeline.

For continuation inside Codex CLI on EC2, use the complete
[EC2 handoff prompt](ec2-codex-handoff.txt), which records the implementation,
verified limits, native setup, commands and acceptance criteria.

## Required behavior

For every accepted MHR mesh, use the subject's recorded sex to choose a garment
from the matching male or female catalog. Choose the garment randomly with a saved
seed. Keep that garment style consistent across three distinct graded sizes:
**tight**, **medium**, and **loose**. Process these jobs before the pipeline advances
to downstream processing for that subject.

## What is ready

- The accepted PLY body is exported to OBJ with unchanged vertices and topology,
  centimetre coordinates, Y up, and feet at Y=0. OBJ is supported by VStitcher's
  imported-avatar workflow.
- Garment selection is deterministic for a fixed seed, subject and catalog.
  Male and female pools are separate, and each chosen garment must offer all
  three fit conditions for that specific body.
- A size is selected using its enrolled native VStitcher size ID, actual size name,
  and measured rest circumference. S/M/L names alone do not determine fit.
- One working copy and a checksum-bound native job are prepared for each variant.
  Original garments and Step 1 body artifacts remain immutable.
- A native CLI script imports the body, selects the actual garment size, prepares
  the garment, requests synchronous uncached dress iterations, finishes, saves a
  new BW scene, and exports both dressed scene and garment-only GLB files.
- Each variant has its own attempt folder and checksums. Completed native exports
  are reusable; missing outputs are retried without regenerating the MHR mesh.

The native script has been tested using an explicit API test double. It has **not**
been run in VStitcher. The separate GUI import pilot below confirms OBJ ingestion,
not native script execution. No clothed dataset meshes are claimed to exist yet.
The two real OBJ avatars and pending plans are saved in
`examples/vstitcher-preparation`; the preparation checks are recorded in
`reports/step-2-preparation-validation.json`.

## Native desktop import pilot

The existing EC2 remote desktop opened VStitcher 2026.2.1 (55315) into an
authenticated workspace. A copy of
`examples/vstitcher-preparation/ansur2_male_10027` was transferred through the
remote desktop clipboard to `C:\Users\Administrator\Desktop\ansur2_male_10027`.
The local-drive redirection path was rejected, so the imported file was the
transferred `avatar.obj` on EC2.

In a fresh garment scene, File > Import loaded this OBJ as an avatar. The MHR body
was visibly rendered upright in its A pose, and Edit Avatar displayed Distance
From Ground = 0 cm. The scene was left open for inspection. The source mesh has
18,439 vertices, 36,874 triangles and a local height of 177.6000061 cm; native
height, topology and a remote checksum have not yet been measured independently.

The editor displayed anchor points, but their anatomical placement has not been
validated. Imported geometry alone does not establish correct garment preparation
or fit. No native CLI/API job, garment dressing, three-size simulation, scene
export or reference VSA save was completed during this pilot. Desktop sign-in
does not establish that a CLI/Open Platform runtime is installed or usable.
See `reports/step-2-native-import-check.json` for the bounded result. The earlier
preparation report remains the record of its original test scope.

## Fit definitions

The initial configurable bands are based on
`ease_ratio = (garment_rest_circumference - body_circumference) / body_circumference`.
The body value is the achieved section perimeter from Step 1, not an unfitted target.

| Requested condition | Minimum ease | Maximum ease, exclusive |
| --- | --- | --- |
| tight | 0% | 6% |
| medium | 6% | 14% |
| loose | 14% | 30% |

These are operational defaults for non-compression clothing, not universal
physical-fit criteria. The selected primary measurement can be chest, waist or
hip circumference. Fabric behavior, other body dimensions, pattern design and
native simulation still determine the observed fit. Labels remain **intended
conditions** until reviewed. No garment patterns or bodies are scaled to manufacture
the size labels. An incomplete size triplet stays pending instead of substituting
an inappropriate size.

## Prepare now

```sh
python -m pipeline run --clothing-config configs/clothing.example.json --clothing-mode prepare --limit 2
```

Or prepare an existing sample without rerunning MHR:

```sh
python -m pipeline clothes --subject-dir examples/fitted/ansur2_male_10027
```

The shipped catalog is intentionally empty. With it, avatar OBJ and `plan.json`
are produced with the three requested conditions and `waiting_for_garment_catalog`
status. They are not counted as completed simulations or delivered downstream.
Once real assets are enrolled, preparation writes `tight/job.json`,
`medium/job.json`, and `loose/job.json` and remains pending VStitcher execution.

## Complete runtime and garment setup

1. The EC2 desktop is accessible. Locate or install the licensed VStitcher-CLI on
   the execution computer and verify its normal StylezoneConnect sign-in and API
   access. No credentials belong in this repository.
2. Copy the example config and catalog to `clothing.local.json` and
   `garment-catalog.local.json`. These local files and `garments/` are ignored by Git.
3. Enroll native `.bw`/`.vsgx` garments for both male and female subjects. For each
   garment supply a stable ID, matching sex, file path, SHA-256, primary measurement,
   measurement provenance, and actual native size IDs/names with measured rest
   circumferences. The schema is `configs/garment-catalog.schema.json`.
4. Set the catalog path, `vstitcher.cli_executable`, timeout and simulation iterations
   in the local config. Paths are resolved relative to the config/catalog file.
   Run on the VStitcher host with these files available locally. This preparation
   does not deploy to EC2 or upload files automatically.
5. Validate a pilot with the generated MHR body: import units, detected avatar
   anchors, pose, garment placement, fabric/gravity settings, sufficient simulation
   iterations, exported scale and fit appearance. If needed, provide a compatible
   single-avatar MHR reference VSA via `avatar_reference_vsa`.
6. Run the same pipeline with `--clothing-mode simulate`. It invokes the licensed
   CLI sequentially for the three variants. Empty catalogs or unavailable runtime
   remain pending/fail; no simulation success is fabricated.

For each executed variant, outputs are `dressed.bw`, `scene.glb` with the avatar,
and `garment.glb` without the avatar, plus a native receipt and logs. Native API
success and valid nonempty exports mean `simulation_completed_unreviewed`;
they do not assert verified physical fit or training-set acceptance.

## Official references

- [Avatar requirements](https://help.browzwear.com/en/articles/13065635-imported-avatar-requirements)
- [VStitcher-CLI installation and execution](https://help.browzwear.com/en/articles/13265438-vstitcher-cli-local-installation-usage-guide)
- [Avatar API](https://help.browzwear.com/en/articles/13066134-avatar)
- [Garment API](https://help.browzwear.com/en/articles/13066118-garment)
- [Size API](https://help.browzwear.com/en/articles/13066127-size)
- [Synchronous simulation options](https://help.browzwear.com/en/articles/14863747-simulation_v3_27-json)
- [Render/export API](https://help.browzwear.com/en/articles/13066137-render)
- [3D export schema](https://help.browzwear.com/en/articles/14863751-export_3d_object_v3_24-json)
