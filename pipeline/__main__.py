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
