"""Sequential, resumable processing of dataset records into downstream handoffs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

RUNNER_VERSION = 1
ARTIFACT_NAMES = ('body.ply', 'mhr-inputs.npz', 'source-record.json', 'fit-report.json')


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def artifact_hashes(directory: Path):
    result = {}
    for name in ARTIFACT_NAMES:
        with (directory / name).open('rb') as stream:
            result[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


@contextmanager
def output_lock(output: Path):
    """OS releases the lock if the program exits or is interrupted."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError(f'A pipeline is already using {output}') from error
        try:
            yield
        finally:
            stream.seek(0)
            release()


def run_pipeline(
    subjects: list[dict], output: Path, fit_subject: Callable,
    verify_subject: Callable, fitting_config: dict,
    next_stage: Callable | None = None, next_stage_id: str | None = None,
):
    """Process each record completely before advancing to the next record.

    A downstream callback receives one handoff dict and returns a JSON receipt.
    Delivery retries use the same idempotency key; consumers must deduplicate it.
    With no callback configured, verified handoff.json files form a durable queue.
    """
    output = output.resolve()
    if next_stage is not None and not next_stage_id:
        raise ValueError('A configured next stage requires a stable stage ID')
    ids = [subject['subject_id'] for subject in subjects]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate subject IDs in this run')
    if any(Path(identifier).name != identifier or identifier in ('.', '..') or '/' in identifier or '\\' in identifier for identifier in ids):
        raise ValueError('Subject IDs must be safe directory names')
    summary = {
        'records_selected': len(subjects), 'generated': 0, 'resumed': 0,
        'ready_for_next_stage': 0, 'delivered': 0, 'rejected': 0, 'failed': 0,
    }

    with output_lock(output):
        events = output / 'events.jsonl'

        def event(subject_id, stage, **details):
            item = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'subject_id': subject_id, 'stage': stage, **details}
            with events.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item, allow_nan=False) + '\n')
            print(json.dumps(item), flush=True)

        write_json(output / 'progress.json', summary)
        results = []
        for subject in subjects:
            subject_id = subject['subject_id']
            directory = output / 'subjects' / subject_id
            checkpoint_path = directory / 'checkpoint.json'
            input_hash = fingerprint({'subject': subject, 'fitting': fitting_config, 'runner_version': RUNNER_VERSION})
            checkpoint = {'input_fingerprint': input_hash, 'subject_id': subject_id, 'mesh_status': 'pending'}
            reused = False
            stage = 'mesh'
            try:
                if checkpoint_path.is_file():
                    try:
                        previous = json.loads(checkpoint_path.read_text(encoding='utf-8'))
                        if previous.get('input_fingerprint') == input_hash and previous.get('mesh_status') == 'verified':
                            if artifact_hashes(directory) != previous['artifact_hashes']:
                                raise ValueError('Saved artifact checksums differ')
                            if verify_subject(directory)['status'] != 'passed':
                                raise ValueError('Saved mesh verification failed')
                            checkpoint = previous
                            reused = True
                    except (OSError, ValueError, KeyError) as error:
                        event(subject_id, 'checkpoint_invalid', error=str(error))

                if reused:
                    summary['resumed'] += 1
                    event(subject_id, 'mesh_resumed')
                else:
                    (directory / 'handoff.json').unlink(missing_ok=True)
                    write_json(checkpoint_path, checkpoint)
                    event(subject_id, 'mesh_started')
                    report = fit_subject(subject, directory)
                    if report['status'] != 'accepted_proxy_fit':
                        checkpoint['mesh_status'] = 'rejected'
                        write_json(checkpoint_path, checkpoint)
                        summary['rejected'] += 1
                        results.append({'subject_id': subject_id, 'status': 'rejected'})
                        event(subject_id, 'mesh_rejected')
                        continue
                    verification = verify_subject(directory)
                    if verification['status'] != 'passed':
                        raise ValueError('Generated mesh verification failed')
                    checkpoint.update({
                        'mesh_status': 'verified', 'artifact_hashes': artifact_hashes(directory),
                        'verification': verification,
                    })
                    write_json(checkpoint_path, checkpoint)
                    summary['generated'] += 1
                    event(subject_id, 'mesh_verified')

                stage = 'handoff'
                artifact_files = {name: (directory / name).relative_to(output).as_posix() for name in ARTIFACT_NAMES}
                mesh_key = fingerprint({'input': input_hash, 'mesh_sha256': checkpoint['artifact_hashes']['body.ply']})
                handoff = {
                    'schema_version': 1, 'subject_id': subject_id,
                    'source_commit': subject['source_commit'], 'stage': 'mhr_mesh',
                    'status': 'ready_for_next_stage', 'units': 'cm', 'vertical_axis': 'Y',
                    'label_kind': 'estimated_MHR_body_from_anthropometry',
                    'output_root': str(output), 'files': artifact_files,
                    'sha256': checkpoint['artifact_hashes'],
                    'idempotency_key': fingerprint({'mesh': mesh_key, 'next_stage': next_stage_id}),
                    'next_stage': next_stage_id,
                }
                write_json(directory / 'handoff.json', handoff)
                if next_stage is None:
                    status = 'ready_for_next_stage'
                    checkpoint.pop('last_error', None)
                    write_json(checkpoint_path, checkpoint)
                    summary[status] += 1
                    event(subject_id, status)
                else:
                    delivery = checkpoint.get('delivery', {})
                    if delivery.get('idempotency_key') == handoff['idempotency_key'] and delivery.get('status') == 'complete':
                        status = 'already_delivered'
                        event(subject_id, status, next_stage=next_stage_id)
                    else:
                        event(subject_id, 'next_stage_started', next_stage=next_stage_id)
                        receipt = next_stage(handoff)
                        checkpoint['delivery'] = {
                            'status': 'complete', 'next_stage': next_stage_id,
                            'idempotency_key': handoff['idempotency_key'], 'receipt': receipt,
                        }
                        checkpoint.pop('last_error', None)
                        write_json(checkpoint_path, checkpoint)
                        status = 'delivered'
                        event(subject_id, status, next_stage=next_stage_id)
                    handoff['status'] = 'delivered'
                    handoff['receipt'] = checkpoint['delivery']['receipt']
                    write_json(directory / 'handoff.json', handoff)
                    summary['delivered'] += 1
                results.append({'subject_id': subject_id, 'status': status})
            except Exception as error:
                summary['failed'] += 1
                checkpoint['last_error'] = {'stage': stage, 'type': type(error).__name__, 'message': str(error)}
                # A failed delivery must retry even if the callback completed its side effect.
                if stage == 'handoff':
                    checkpoint.pop('delivery', None)
                write_json(checkpoint_path, checkpoint)
                results.append({'subject_id': subject_id, 'status': 'failed', 'stage': stage, 'error': str(error)})
                event(subject_id, 'failed', failed_stage=stage, error=str(error))
            finally:
                write_json(output / 'progress.json', summary)
        report = {**summary, 'results': results}
        write_json(output / 'run-report.json', report)
        return report
