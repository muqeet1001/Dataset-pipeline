"""Lock one output job and verify reusable artifacts before accepting a receipt."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys


@contextmanager
def job_lock(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    stream=(output/'.job.lock').open('a+b')
    if stream.tell()==0:
        stream.write(b'0'); stream.flush()
    stream.seek(0)
    try:
        if sys.platform=='win32':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError('Another clothing job is using this output folder') from error
    try:
        yield
    finally:
        stream.seek(0)
        if sys.platform=='win32':
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def reusable(report_path, signature):
    try:
        path=Path(report_path)
        report=json.loads(path.read_text())
        if report.get('signature')!=signature or report.get('status')!='numerical_pass_unvalidated_fabric':
            return None
        if not report.get('artifacts'):
            return None
        for name,expected in report['artifacts'].items():
            if Path(name).name!=name:
                return None
            actual=hashlib.sha256((path.parent/name).read_bytes()).hexdigest()
            if actual!=expected:
                return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None
