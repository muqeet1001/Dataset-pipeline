import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Dataset-pipeline')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('smoke-mhr')
    command.add_argument('--model', type=Path, default=Path('assets/mhr_model.pt'))
    command.add_argument('--output', type=Path, default=Path('runs/smoke'))
    command = commands.add_parser('normalize')
    command.add_argument('--data', type=Path, default=Path('data/raw/ansur2'))
    command.add_argument('--output', type=Path, default=Path('data/processed/subjects.jsonl'))
    command = commands.add_parser('fit')
    command.add_argument('--model', type=Path, default=Path('assets/mhr_model.pt'))
    command.add_argument('--data', type=Path, default=Path('data/raw/ansur2'))
    command.add_argument('--output', type=Path, default=Path('runs/subjects'))
    selection = command.add_mutually_exclusive_group(required=True)
    selection.add_argument('--subject-id', action='append')
    selection.add_argument('--all', action='store_true')
    command.add_argument('--max-error-cm', type=float, default=2.0)
    command.add_argument('--max-evaluations', type=int, default=150)
    command = commands.add_parser('verify')
    command.add_argument('--model', type=Path, default=Path('assets/mhr_model.pt'))
    command.add_argument('--directory', type=Path, default=Path('examples/fitted'))
    command = commands.add_parser('run', help='Process dataset records sequentially and resume completed records')
    command.add_argument('--model', type=Path, default=Path('assets/mhr_model.pt'))
    command.add_argument('--data', type=Path, default=Path('data/raw/ansur2'))
    command.add_argument('--output', type=Path, default=Path('runs/pipeline'))
    selection = command.add_mutually_exclusive_group()
    selection.add_argument('--subject-id', action='append')
    selection.add_argument('--limit', type=int, help='Process only the first N records for a trial run')
    command.add_argument('--max-error-cm', type=float, default=2.0)
    command.add_argument('--max-evaluations', type=int, default=150)
    command.add_argument('--next-stage', metavar='MODULE:FUNCTION', help='Call this local Python function for every verified mesh')
    command.add_argument('--clothing-config', type=Path, help='Connect the VStitcher clothing stage')
    command.add_argument('--clothing-mode', choices=('prepare', 'simulate'), default='prepare')
    command = commands.add_parser('clothes', help='Prepare clothing for an existing accepted subject mesh')
    command.add_argument('--subject-dir', type=Path, required=True)
    command.add_argument('--config', type=Path, default=Path('configs/clothing.example.json'))
    command.add_argument('--output', type=Path, default=Path('runs/clothing'))
    command.add_argument('--mode', choices=('prepare', 'simulate'), default='prepare')
    args = parser.parse_args()
    if args.command == 'normalize':
        from .dataset import normalize
        print(json.dumps(normalize(args.data, args.output), indent=2))
    elif args.command == 'smoke-mhr':
        from .mhr import smoke_test
        print(json.dumps(smoke_test(args.model, args.output), indent=2))
    elif args.command == 'verify':
        from .mhr import load_model
        from .verify import verify_generated
        print(json.dumps(verify_generated(load_model(args.model), args.directory), indent=2))
    elif args.command == 'clothes':
        from .clothing import ClothingStage, handoff_from_directory
        result = ClothingStage(args.config, args.mode, args.output)(handoff_from_directory(args.subject_dir))
        print(json.dumps(result, indent=2))
        if args.mode == 'simulate' and result.get('status') == 'pending':
            raise SystemExit(2)
    elif args.command == 'run':
        import hashlib
        import importlib
        import importlib.metadata
        import math
        from .dataset import load_subjects
        from .fit import BodyFitter
        from .mhr import load_model
        from .runner import run_pipeline
        from .verify import verify_subject
        if not math.isfinite(args.max_error_cm) or args.max_error_cm <= 0 or args.max_evaluations < 1:
            parser.error('Error tolerance and maximum evaluations must be positive and finite')
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
        subjects = load_subjects(args.data)
        if args.subject_id:
            requested = set(args.subject_id)
            subjects = [subject for subject in subjects if subject['subject_id'] in requested]
            missing = requested - {subject['subject_id'] for subject in subjects}
            if missing:
                parser.error(f'Unknown subject IDs: {sorted(missing)}')
        elif args.limit:
            subjects = subjects[:args.limit]
        next_stage = None
        next_stage_id = args.next_stage
        if args.next_stage and args.clothing_config:
            parser.error('Choose --next-stage or --clothing-config, not both')
        if args.clothing_config:
            from .clothing import ClothingStage
            next_stage = ClothingStage(args.clothing_config, args.clothing_mode)
            next_stage_id = next_stage.stage_id
        if args.next_stage:
            module_name, separator, function_name = args.next_stage.partition(':')
            if not separator or not module_name or not function_name or ':' in function_name:
                parser.error('--next-stage must be MODULE:FUNCTION')
            try:
                next_stage = getattr(importlib.import_module(module_name), function_name)
            except (ImportError, AttributeError) as error:
                parser.error(f'Cannot load next stage: {error}')
            if not callable(next_stage):
                parser.error('--next-stage must name a callable function')
        model = load_model(args.model)
        fitter = BodyFitter(model, args.max_error_cm)
        source_root = Path(__file__).parent
        provenance = json.loads((source_root.parent / 'third_party/mhr-provenance.json').read_text(encoding='utf-8'))
        config = {
            'model_sha256': provenance['model_sha256'],
            'max_error_cm': args.max_error_cm, 'max_evaluations': args.max_evaluations,
            'code_sha256': {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in ('dataset.py', 'fit.py', 'sections.py', 'mhr.py', 'verify.py')},
            'dependencies': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy', 'trimesh')},
        }
        report = run_pipeline(
            subjects, args.output,
            lambda subject, directory: fitter.fit(subject, directory, args.max_evaluations),
            lambda directory: verify_subject(model, directory),
            fitting_config=config, next_stage=next_stage, next_stage_id=next_stage_id,
        )
        print(json.dumps({key: value for key, value in report.items() if key != 'results'}, indent=2))
        if report['rejected'] or report['failed'] or (args.clothing_config and args.clothing_mode == 'simulate' and report['pending_next_stage']):
            raise SystemExit(2)
    else:
        from .dataset import load_subjects
        from .fit import BodyFitter
        from .mhr import load_model
        subjects = load_subjects(args.data)
        if args.subject_id:
            requested = set(args.subject_id)
            subjects = [subject for subject in subjects if subject['subject_id'] in requested]
            missing = requested - {subject['subject_id'] for subject in subjects}
            if missing:
                parser.error(f'Unknown subject IDs: {sorted(missing)}')
        if args.max_error_cm <= 0 or args.max_evaluations < 1:
            parser.error('Error tolerance and maximum evaluations must be positive')
        fitter = BodyFitter(load_model(args.model), args.max_error_cm)
        results = []
        for subject in subjects:
            report = fitter.fit(subject, args.output / subject['subject_id'], args.max_evaluations)
            result = {key: report[key] for key in ('subject_id', 'status', 'elapsed_seconds')}
            results.append(result)
            print(json.dumps(result), flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'batch-report.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        if any(result['status'] != 'accepted_proxy_fit' for result in results):
            raise SystemExit(2)


if __name__ == '__main__':
    main()
