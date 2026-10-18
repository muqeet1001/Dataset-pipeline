import json
from pathlib import Path
import pytest
from pipeline.runner import ARTIFACT_NAMES, output_lock, run_pipeline


def subjects():
    return [{'subject_id': name, 'source_commit': 'source-commit'} for name in ('subject_a', 'subject_b')]


class StageDouble:
    def __init__(self):
        self.calls = []
        self.rejected = set()
        self.errors = set()

    def fit(self, subject, directory):
        name = subject['subject_id']
        self.calls.append(('fit', name))
        if name in self.errors:
            raise RuntimeError('Mesh stage failed')
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ARTIFACT_NAMES:
            (directory / filename).write_text(name)
        return {'status': 'rejected_fit' if name in self.rejected else 'accepted_proxy_fit'}

    def verify(self, directory):
        self.calls.append(('verify', directory.name))
        assert (directory / 'body.ply').is_file()
        return {'status': 'passed'}

    def consume(self, handoff):
        self.calls.append(('next', handoff['subject_id']))
        assert (Path(handoff['output_root']) / handoff['files']['body.ply']).is_file()
        return {'completed': True}


def execute(tmp_path, double, selected=None, **options):
    return run_pipeline(
        selected if selected is not None else subjects(), tmp_path,
        double.fit, double.verify, options.pop('fitting_config', {'version': 1}), **options,
    )


def test_processes_each_record_through_next_stage_before_starting_next(tmp_path):
    stage = StageDouble()
    report = execute(tmp_path, stage, next_stage=stage.consume, next_stage_id='consumer:v1')
    assert stage.calls == [
        ('fit', 'subject_a'), ('verify', 'subject_a'), ('next', 'subject_a'),
        ('fit', 'subject_b'), ('verify', 'subject_b'), ('next', 'subject_b'),
    ]
    assert report['generated'] == 2 and report['delivered'] == 2
    assert json.loads((tmp_path / 'subjects/subject_a/handoff.json').read_text())['status'] == 'delivered'


def test_resume_does_not_regenerate_or_redeliver_completed_records(tmp_path):
    stage = StageDouble()
    execute(tmp_path, stage, next_stage=stage.consume, next_stage_id='consumer:v1')
    stage.calls.clear()
    report = execute(tmp_path, stage, next_stage=stage.consume, next_stage_id='consumer:v1')
    assert stage.calls == [('verify', 'subject_a'), ('verify', 'subject_b')]
    assert report['generated'] == 0 and report['resumed'] == 2
    assert all(result['status'] == 'already_delivered' for result in report['results'])


def test_default_writes_ready_handoffs_without_claiming_delivery(tmp_path):
    stage = StageDouble()
    report = execute(tmp_path, stage)
    handoff = json.loads((tmp_path / 'subjects/subject_a/handoff.json').read_text())
    assert handoff['status'] == 'ready_for_next_stage'
    assert handoff['next_stage'] is None
    assert report['ready_for_next_stage'] == 2 and report['delivered'] == 0


def test_rejected_and_failed_meshes_do_not_reach_next_stage(tmp_path):
    stage = StageDouble()
    stage.rejected.add('subject_a')
    stage.errors.add('subject_b')
    report = execute(tmp_path, stage, next_stage=stage.consume, next_stage_id='consumer:v1')
    assert not any(call[0] == 'next' for call in stage.calls)
    assert report['rejected'] == 1 and report['failed'] == 1
    assert not list(tmp_path.glob('subjects/*/handoff.json'))


def test_delivery_retry_reuses_mesh_and_idempotency_key(tmp_path):
    stage = StageDouble()
    attempts = []

    def consumer(handoff):
        attempts.append(handoff['idempotency_key'])
        if len(attempts) == 1:
            raise RuntimeError('Consumer interrupted after receiving the mesh')
        return {'completed': True}

    selected = subjects()[:1]
    first = execute(tmp_path, stage, selected, next_stage=consumer, next_stage_id='consumer:v1')
    second = execute(tmp_path, stage, selected, next_stage=consumer, next_stage_id='consumer:v1')
    assert first['failed'] == 1 and second['delivered'] == 1
    assert second['generated'] == 0 and second['resumed'] == 1
    assert len(attempts) == 2 and attempts[0] == attempts[1]


def test_corrupted_artifact_is_regenerated(tmp_path):
    stage = StageDouble()
    execute(tmp_path, stage, subjects()[:1])
    (tmp_path / 'subjects/subject_a/body.ply').write_text('corrupted')
    report = execute(tmp_path, stage, subjects()[:1])
    assert report['generated'] == 1 and report['resumed'] == 0
    assert 'checkpoint_invalid' in (tmp_path / 'events.jsonl').read_text()


def test_changed_configuration_regenerates_and_withdraws_stale_handoff(tmp_path):
    stage = StageDouble()
    execute(tmp_path, stage, subjects()[:1])
    stage.rejected.add('subject_a')
    report = execute(tmp_path, stage, subjects()[:1], fitting_config={'version': 2})
    assert report['rejected'] == 1
    assert not (tmp_path / 'subjects/subject_a/handoff.json').exists()


def test_new_stage_can_consume_existing_verified_meshes(tmp_path):
    stage = StageDouble()
    execute(tmp_path, stage)
    stage.calls.clear()
    report = execute(tmp_path, stage, next_stage=stage.consume, next_stage_id='consumer:v1')
    assert report['generated'] == 0 and report['delivered'] == 2
    assert sum(call[0] == 'next' for call in stage.calls) == 2


def test_output_lock_releases_after_interruption(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with output_lock(tmp_path):
            with pytest.raises(RuntimeError, match='already using'):
                with output_lock(tmp_path):
                    pytest.fail('Concurrent run was allowed')
            raise KeyboardInterrupt
    with output_lock(tmp_path):
        pass


def test_interrupt_then_resume_keeps_completed_record(tmp_path):
    stage = StageDouble()
    ordinary_fit = stage.fit

    def interrupted_fit(subject, directory):
        if subject['subject_id'] == 'subject_b':
            raise KeyboardInterrupt
        return ordinary_fit(subject, directory)

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(subjects(), tmp_path, interrupted_fit, stage.verify, {'version': 1})
    stage.calls.clear()
    report = execute(tmp_path, stage)
    assert report['resumed'] == 1 and report['generated'] == 1
    assert ('fit', 'subject_a') not in stage.calls


def test_pending_next_stage_is_not_recorded_as_delivered(tmp_path):
    stage = StageDouble()
    pending = lambda handoff: {'status': 'pending', 'reason': 'waiting_for_access'}
    first = execute(tmp_path, stage, subjects()[:1], next_stage=pending, next_stage_id='pending:v1')
    assert first['pending_next_stage'] == 1 and first['delivered'] == 0
    checkpoint = json.loads((tmp_path / 'subjects/subject_a/checkpoint.json').read_text())
    assert checkpoint['delivery']['status'] == 'pending'
    second = execute(tmp_path, stage, subjects()[:1], next_stage=stage.consume, next_stage_id='pending:v1')
    assert second['resumed'] == 1 and second['delivered'] == 1


def test_missing_downstream_outputs_are_reprocessed_without_new_mesh(tmp_path):
    stage = StageDouble()

    class Consumer:
        valid = True

        def __call__(self, handoff):
            self.valid = True
            return stage.consume(handoff)

        def validate_receipt(self, receipt):
            return self.valid

    consumer = Consumer()
    execute(tmp_path, stage, subjects()[:1], next_stage=consumer, next_stage_id='validated:v1')
    consumer.valid = False
    stage.calls.clear()
    second = execute(tmp_path, stage, subjects()[:1], next_stage=consumer, next_stage_id='validated:v1')
    assert second['resumed'] == 1 and second['delivered'] == 1
    assert stage.calls == [('verify', 'subject_a'), ('next', 'subject_a')]
