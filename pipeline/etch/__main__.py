"""Explicit commands for the MHR-native ETCH method adaptation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    markers=sub.add_parser('markers')
    markers.add_argument('--input',type=Path,default=ROOT/'inputs/MHR-male-10027')
    markers.add_argument('--output',type=Path,default=ROOT/'configs/mhr-markers-86.json')
    markers.add_argument('--force',action='store_true')
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--input',type=Path,default=ROOT/'inputs/MHR-male-10027')
    prepare.add_argument('--fit-folder',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--points',type=int,default=5000)
    prepare.add_argument('--anchors',type=int,default=6000)
    prepare.add_argument('--markers',type=Path,default=ROOT/'configs/mhr-markers-86.json')
    prepare.add_argument('--development',action='store_true')
    train=sub.add_parser('train')
    train.add_argument('--pairs',type=Path,default=ROOT/'runs/etch/pairs')
    train.add_argument('--output',type=Path,default=ROOT/'runs/etch/checkpoints/native.pt')
    train.add_argument('--markers',type=Path,default=ROOT/'configs/mhr-markers-86.json')
    train.add_argument('--steps',type=int,default=1000)
    train.add_argument('--points',type=int,default=5000)
    train.add_argument('--width',type=int,default=32)
    train.add_argument('--device',default='cpu')
    train.add_argument('--development',action='store_true')
    predict=sub.add_parser('predict')
    predict.add_argument('--scan',type=Path,required=True)
    predict.add_argument('--checkpoint',type=Path,required=True)
    predict.add_argument('--output',type=Path,default=ROOT/'runs/etch/prediction')
    predict.add_argument('--points',type=int,default=5000)
    predict.add_argument('--device',default='cpu')
    predict.add_argument('--mhr-model',type=Path,default=ROOT/'assets/mhr_model.pt')
    predict.add_argument('--no-fit-body',action='store_true')
    predict.add_argument('--development',action='store_true')
    oracle=sub.add_parser('oracle-check',help='Test the body fitter with known ground-truth markers; NOT neural inference')
    oracle.add_argument('--input',type=Path,default=ROOT/'inputs/MHR-male-10027')
    oracle.add_argument('--markers',type=Path,default=ROOT/'configs/mhr-markers-86.json')
    oracle.add_argument('--mhr-model',type=Path,default=ROOT/'assets/mhr_model.pt')
    oracle.add_argument('--output',type=Path,default=ROOT/'runs/etch/oracle-check')
    args=parser.parse_args(argv)
    from ..avatar_clothing import load_body,write_json
    if args.command=='doctor':
        import importlib.metadata as metadata
        import torch
        from . import UPSTREAM_COMMIT
        print(json.dumps({'upstream_commit_reviewed':UPSTREAM_COMMIT,
            'cuda_available':torch.cuda.is_available(),
            'versions':{p:metadata.version(p) for p in ('torch','numpy','scipy','trimesh','potpourri3d')},
            'official_etch_weights_used':False,'official_etch_cuda_extensions_used':False,
            'architecture':'independent MHR-native ETCH-task adaptation',
            'mhr_asset_exists':(ROOT/'assets/mhr_model.pt').exists(),
            'articulated_body_fitting_implemented':False},indent=2))
        return 0
    if args.command=='markers':
        from .geometry import create_markers
        if args.output.exists() and not args.force:
            print('Keeping existing marker definition; all subjects must share these IDs')
            return 0
        body,_,_=load_body(args.input);write_json(args.output,create_markers(body))
        print(args.output);return 0
    if args.command=='prepare':
        from .prepare import prepare_pair
        if args.output.suffix!='.npz':parser.error('--output must end with .npz')
        if min(args.points,args.anchors)<1:parser.error('Points and anchors must be positive')
        result=prepare_pair(args.input,args.fit_folder,args.markers,args.output,
            count=args.points,anchor_count=args.anchors,development=args.development)
        print(json.dumps(result,indent=2));return 0
    if args.command=='train':
        from .runtime import train
        paths=[args.pairs] if args.pairs.is_file() else sorted(args.pairs.glob('*.npz'))
        report=train(paths,args.markers,args.output,args.steps,args.points,args.width,args.device,args.development)
        print(json.dumps({k:v for k,v in report.items() if k not in ('history','marker_definition')},indent=2))
        return 0
    if args.command=='predict':
        from .runtime import predict
        markers,valid,definition,report=predict(args.scan,args.checkpoint,args.output,
            args.points,device=args.device,development=args.development)
        if not args.no_fit_body:
            from ..mhr import load_model
            from .body_fit import MHRMarkerFitter
            try:
                if valid.sum()<24:
                    raise ValueError(f'Only {int(valid.sum())} marker labels were predicted; at least 24 are required')
                fitter=MHRMarkerFitter(load_model(args.mhr_model,threads=2),definition)
                mesh,fit,inputs=fitter.fit(markers,valid)
                mesh.export(args.output/'predicted-mhr.obj')
                np.savez_compressed(args.output/'mhr-inputs.npz',**inputs)
                write_json(args.output/'body-fit-report.json',fit)
                report['body_fit_status']=fit['status']
            except ValueError as error:
                report['body_fit_status']='failed';report['body_fit_error']=str(error)
        write_json(args.output/'prediction-report.json',report)
        print(json.dumps(report,indent=2))
        return 2 if report.get('body_fit_status') in ('failed','needs_review') else 0
    if args.command=='oracle-check':
        from ..mhr import load_model
        from .body_fit import MHRMarkerFitter
        body,_,_=load_body(args.input);definition=json.loads(args.markers.read_text())
        print('Testing official MHR fitter against known markers (not neural inference)...',flush=True)
        fitter=MHRMarkerFitter(load_model(args.mhr_model,threads=2),definition)
        mesh,report,inputs=fitter.fit(body.vertices[definition['vertex_indices']])
        report.update({'source':'GROUND-TRUTH MARKER ORACLE; NOT NEURAL INFERENCE',
            'neural_accuracy_evaluated':False,
            'all_vertex_rmse_m':float(np.sqrt(np.mean(np.sum((mesh.vertices-body.vertices)**2,axis=1)))),
            'all_vertex_max_error_m':float(np.linalg.norm(mesh.vertices-body.vertices,axis=1).max())})
        args.output.mkdir(parents=True,exist_ok=True);mesh.export(args.output/'body.obj')
        np.savez_compressed(args.output/'mhr-inputs.npz',**inputs)
        write_json(args.output/'oracle-report.json',report);print(json.dumps(report,indent=2))
        return 0 if report['status']=='neutral_marker_fit_passed' else 2
    return 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError,RuntimeError) as error:
        print(f'ETCH adaptation: {error}',file=sys.stderr);raise SystemExit(2)
