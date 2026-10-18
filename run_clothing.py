"""One-command setup, cloth run and local viewing for the supplied avatar."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import shutil
import sys
import time
import urllib.error
import urllib.request
import venv
import webbrowser

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view-only',action='store_true')
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--steps',type=int,default=150)
    parser.add_argument('--port',type=int,default=9086)
    args=parser.parse_args()
    python=ROOT/'.venv'/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not python.exists():
        venv.create(ROOT/'.venv',with_pip=True)
    requirements=ROOT/'requirements-cloth.txt'
    expected=hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp=ROOT/'.venv/cloth-setup.sha256'
    if not stamp.exists() or stamp.read_text().strip()!=expected:
        subprocess.check_call([str(python),'-m','pip','install','-r',str(requirements)],cwd=ROOT)
        stamp.write_text(expected,encoding='ascii')
    viewer=ROOT/'web/clothing-viewer'
    if not (viewer/'node_modules/three/package.json').exists():
        subprocess.check_call(['npm.cmd' if sys.platform=='win32' else 'npm','ci','--ignore-scripts','--no-audit','--no-fund'],cwd=viewer)
    saved=ROOT/'runs/local-avatar/MHR-male-10027'
    example=ROOT/'examples/clothing-pilot/MHR-male-10027'
    if not saved.exists() and (example/'manifest.json').exists():
        shutil.copytree(example,saved)
    url=f'http://127.0.0.1:{args.port}/web/clothing-viewer/'
    try:
        with urllib.request.urlopen(url,timeout=2) as response:
            if b'MHR cloth fitting' not in response.read():
                raise RuntimeError('Viewer port belongs to a different local service')
    except urllib.error.URLError:
        (ROOT/'work').mkdir(exist_ok=True)
        log=(ROOT/'work/viewer.log').open('a',encoding='utf-8')
        process=subprocess.Popen([str(python),'-u',str(ROOT/'scripts/serve_clothing.py'),'--port',str(args.port)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0,
            start_new_session=sys.platform!='win32')
        (ROOT/'work/viewer.pid').write_text(str(process.pid))
        for _ in range(40):
            try:
                with urllib.request.urlopen(url,timeout=.5) as response:
                    if response.status==200:
                        break
            except urllib.error.URLError:
                time.sleep(.1)
        else:
            raise RuntimeError('The local viewer did not start; inspect work/viewer.log')
    print(f'Open clothing viewer: {url}')
    if not args.no_browser:
        webbrowser.open(url)
    code=0
    if not args.view_only:
        code=subprocess.call([str(python),'-u','-m','pipeline.avatar_clothing','--device',args.device,
            '--steps',str(args.steps),'--substeps','4','--iterations','8','--face-budget','1200'],cwd=ROOT)
        if code not in (0,2):
            print('Cloth run failed. Inspect the terminal and saved manifest before retrying.')
            return code
    if code==2:
        print('Outputs are viewable; some numerical checks need review. This is not a training-ready result.')
    return code


if __name__=='__main__':
    raise SystemExit(main())
