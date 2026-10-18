import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Dataset-pipeline')
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('smoke-mhr')
    command.add_argument('--model', type=Path, default=Path('assets/mhr_model.pt'))
    command.add_argument('--output', type=Path, default=Path('runs/smoke'))
    args = parser.parse_args()
    from .mhr import smoke_test
    print(json.dumps(smoke_test(args.model, args.output), indent=2))


if __name__ == '__main__':
    main()
