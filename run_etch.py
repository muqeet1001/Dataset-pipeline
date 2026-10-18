"""Entry point: python run_etch.py --help."""
from pipeline.etch.__main__ import main

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,FileNotFoundError,RuntimeError) as error:
        print(f'ETCH adaptation: {error}')
        raise SystemExit(2)
