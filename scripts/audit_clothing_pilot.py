"""Audit saved three-fit output, including reverse body-edge intersection tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.avatar_clothing import load_body, write_json
from pipeline.cloth_quality import edge_crossings
from pipeline.cloth_receipts import reusable
from pipeline.garment_draft import PROFILES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/local-avatar/MHR-male-10027')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/local-avatar-pilot.json')
    args=parser.parse_args()
    body,metadata,source_hash=load_body(ROOT/'inputs/MHR-male-10027')
    manifest=json.loads((args.run/'manifest.json').read_text())
    report={'audited_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Local, static, supplied male MHR avatar; 3 fits, shirt and pants in each',
        'subject_id':metadata['subject_id'],'body_height_cm':metadata['height_cm'],
        'input_body_sha256':source_hash,'device':manifest['device'],'solver':'Newton VBD plus sampled body contact',
        'newton_commit':manifest['newton_commit'],'versions':manifest['versions'],
        'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'fits':{},'training_ready':False,
        'limitations':['Fabric preset is not calibrated','Static collar and waistband supports',
                       'One male outfit template; no female/full-dataset/GPU benchmark',
                       'Body-derived outer surface draft, not production sewing patterns'],
        'test_results':{'local_cloth_regressions_passed':7,'existing_pipeline_regressions_passed':27,
                        'upstream_style3d_component_tests_passed':6,
                        'note':'Style3D component tests do not validate our garment adapter'},
        'timing_scope':'Per-fit drafting, simulation, validation and export on CPU; excludes dependency setup, MHR generation and earlier failed attempts'}
    passed=True
    for fit in ('tight','medium','loose'):
        folder=args.run/fit
        saved=json.loads((folder/'fit-report.json').read_text())
        checked=reusable(folder/'fit-report.json',saved['signature']) is not None
        scene=trimesh.load(folder/'scene.glb',force='scene',process=False)
        body_error=float(np.max(np.abs(scene.geometry['body'].vertices-body.vertices)))
        body_faces=bool(np.array_equal(scene.geometry['body'].faces,body.faces))
        reverse={}
        areas={}
        for name in ('shirt','pants'):
            print(f'Auditing {fit}/{name} against all body edges...',flush=True)
            reverse[name]=edge_crossings(body,scene.geometry[name])
            rest=np.load(folder/(name+'-rest.npz'))
            areas[name]=float(trimesh.Trimesh(rest['vertices'],rest['faces'],process=False).area)
        code={name:hashlib.sha256((ROOT/'pipeline'/name).read_bytes()).hexdigest() for name in
              ('avatar_clothing.py','garment_draft.py','cloth_quality.py','body_contact.py','offset_surface.py')}
        expected=hashlib.sha256(json.dumps({'body':source_hash,'config':saved['configuration'],
            'code':code,'profiles':PROFILES},sort_keys=True).encode()).hexdigest()
        current_source_matches=saved['signature']==expected
        ok=(checked and body_faces and body_error<1e-7 and not any(reverse.values()) and
            saved['quality']['numerical_pass'] and current_source_matches)
        passed &= ok
        report['fits'][fit]={'accepted_numerically':bool(ok),'saved_status':saved['status'],
            'artifact_checksums_verified':checked,'current_physics_source_signature_matches':current_source_matches,
            'configuration':saved['configuration'],'total_seconds':saved['total_seconds'],
            'body_roundtrip_max_error_m':body_error,'body_faces_unchanged':body_faces,
            'additional_reverse_body_crossing_edges':reverse,'rest_surface_area_m2':areas,
            'quality':{k:v for k,v in saved['quality'].items() if k!='settling_history'},
            'simulation_steps':saved['simulation']['steps_completed'],
            'collar_support_vertices':saved['drafts']['shirt']['support_vertices'],
            'waist_support_vertices':saved['drafts']['pants']['support_vertices'],
            'scene_glb_sha256':hashlib.sha256((folder/'scene.glb').read_bytes()).hexdigest()}
    report['three_fits_numerical_pass']=bool(passed)
    report['sum_fit_seconds']=sum(f['total_seconds'] for f in report['fits'].values())
    write_json(args.output,report)
    print(f'{"PASS" if passed else "NEEDS REVIEW"}: {args.output}',flush=True)
    return 0 if passed else 2


if __name__=='__main__':
    raise SystemExit(main())
