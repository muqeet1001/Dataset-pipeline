# Third-party sources

`third_party/MHR` contains the complete, unmodified tracked source of Meta's
Momentum Human Rig from `https://github.com/facebookresearch/MHR` at commit
`d96fafa33bbf018647c70c3525e91f53e79d2a14`.

Its Apache-2.0 license, copyright notices, README, demo, tests, tools and viewer
are preserved. See `third_party/MHR/LICENSE` and `third_party/mhr-provenance.json`.

The official release assets are obtained separately from release `v1.0.1`.
They are excluded from Git because of their size. The expected archive SHA-256
and official download URL are recorded in the provenance file.

ANSUR II data provenance is recorded in `data/raw/ansur2/SOURCE.md` and
`data/raw/ansur2/provenance.json`.

The local cloth prototype installs Newton at commit
`f49482ce07e20764110fd147a67c7334eeae638d` (Apache-2.0) and modern NVIDIA Warp
1.18.0. It does not install the research-restricted GarmentCode Warp fork or
CCCollisions. The viewer installs Three.js 0.186.1 (MIT) through npm. Runtime
versions are pinned in `requirements-cloth.txt` and the viewer package lock;
installed distributions retain their own license files. See
`reports/cloth-source-review.json` for upstream code references and
`reports/local-avatar-pilot.json` for actual local evidence.

The generated prototype garment templates are original body-derived drafts.
No third-party garment asset pack or trained cloth checkpoint was incorporated.
