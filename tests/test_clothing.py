import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from pipeline.clothing import ClothingStage, digest, handoff_from_directory, read_config, select_sizes
from pipeline.vstitcher_job import run_job

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def library(tmp_path):
    """Synthetic catalog/files test selection and transport, never native cloth physics."""
    config = json.loads((ROOT / 'configs/clothing.example.json').read_text())
    config['catalog'] = 'catalog.json'
    garments = []
    for sex in ('male', 'female'):
        for number in range(2):
            path = tmp_path / f'{sex}-{number}.bw'
            path.write_bytes(b'SYNTHETIC TEST FILE; NOT A REAL BW GARMENT')
            garments.append({
                'id': f'{sex}-{number}', 'sex': sex, 'file': path.name,
                'sha256': digest(path), 'measurement': 'chestcircumference',
                'measurement_source': 'synthetic test fixture only',
                'sizes': [
                    {'native_size_id': 10, 'name': 'test-tight', 'circumference_cm': 103},
                    {'native_size_id': 20, 'name': 'test-medium', 'circumference_cm': 110},
                    {'native_size_id': 30, 'name': 'test-loose', 'circumference_cm': 120},
                ],
            })
    catalog = {'schema_version': 1, 'garments': garments}
    path = tmp_path / 'clothing.json'
    path.write_text(json.dumps(config))
    (tmp_path / 'catalog.json').write_text(json.dumps(catalog))
    return path, config, catalog


def body(tmp_path, sex='male'):
    directory = tmp_path / ('body-' + sex)
    directory.mkdir(exist_ok=True)
    mesh = trimesh.creation.box(extents=(40, 170, 20))
    mesh.apply_translation([0, 85, 0])
    mesh.export(directory / 'body.ply')
    np.savez(directory / 'mhr-inputs.npz', fixture=np.zeros(1))
    subject = {'subject_id': 'test_' + sex, 'sex': sex, 'source_commit': 'synthetic-test-source'}
    (directory / 'source-record.json').write_text(json.dumps(subject))
    report = {
        'subject_id': subject['subject_id'], 'status': 'accepted_proxy_fit',
        'mesh_sha256': digest(directory / 'body.ply'), 'height_achieved_cm': 170,
        'measurements': {'chestcircumference': {'achieved_cm': 100}},
    }
    (directory / 'fit-report.json').write_text(json.dumps(report))
    return directory, subject, report


def test_seeded_selection_preserves_sex_and_three_distinct_sizes(library):
    _, config, catalog = library
    for sex in ('male', 'female'):
        subject = {'subject_id': 'example_' + sex, 'sex': sex}
        report = {'measurements': {'chestcircumference': {'achieved_cm': 100}}}
        chosen, reason = select_sizes(subject, report, config, catalog)
        assert reason is None and chosen['garment']['sex'] == sex
        assert [size['fit'] for size in chosen['variants']] == ['tight', 'medium', 'loose']
        assert [size['native_size_id'] for size in chosen['variants']] == [10, 20, 30]
        assert select_sizes(subject, report, config, catalog)[0] == chosen
    selections = {select_sizes({'subject_id': str(i), 'sex': 'male'}, report, config, catalog)[0]['garment']['id'] for i in range(20)}
    assert len(selections) == 2


def test_missing_size_band_blocks_instead_of_relabelling(library):
    _, config, catalog = library
    catalog = copy.deepcopy(catalog)
    for garment in catalog['garments']:
        garment['sizes'] = garment['sizes'][:2]
    chosen, reason = select_sizes({'subject_id': 'x', 'sex': 'male'}, {'measurements': {'chestcircumference': {'achieved_cm': 100}}}, config, catalog)
    assert chosen is None and reason == 'no_garment_with_three_matching_sizes'


def test_preparation_exports_unchanged_avatar_and_three_jobs(library, tmp_path):
    config_path, _, _ = library
    directory, _, _ = body(tmp_path)
    handoff = handoff_from_directory(directory)
    result = ClothingStage(config_path, output=tmp_path / 'prepared')(handoff)
    assert result['status'] == 'pending' and result['variants_prepared'] == 3
    plan_path = Path(result['plan'])
    plan = json.loads(plan_path.read_text())
    assert not plan['native_simulation_executed']
    assert plan['status'] == 'prepared_not_executed'
    assert len({json.loads((plan_path.parent / job['file']).read_text())['garment_id'] for job in plan['jobs']}) == 1
    original = trimesh.load_mesh(directory / 'body.ply', process=False)
    exported = trimesh.load_mesh(plan_path.parent / 'avatar.obj', process=False)
    np.testing.assert_allclose(original.vertices, exported.vertices, atol=1e-7)


def test_empty_catalog_keeps_waiting_without_claiming_simulation(tmp_path):
    directory, _, _ = body(tmp_path)
    result = ClothingStage(ROOT / 'configs/clothing.example.json', output=tmp_path / 'prepared')(handoff_from_directory(directory))
    assert result['status'] == 'pending' and result['variants_prepared'] == 0
    assert result['reason'] == 'missing_sex_matched_garments'


def test_missing_cli_cannot_simulate(library, tmp_path):
    directory, _, _ = body(tmp_path)
    with pytest.raises(RuntimeError, match='access is not configured'):
        ClothingStage(library[0], mode='simulate', output=tmp_path / 'prepared')(handoff_from_directory(directory))


def test_mutated_source_garment_is_rejected(library, tmp_path):
    config_path, _, catalog = library
    directory, _, _ = body(tmp_path)
    for garment in catalog['garments']:
        (config_path.parent / garment['file']).write_bytes(b'modified')
    with pytest.raises(ValueError, match='checksum differs'):
        ClothingStage(config_path, output=tmp_path / 'prepared')(handoff_from_directory(directory))


def test_overlapping_fit_bands_are_rejected(library):
    config_path, config, _ = library
    config['fit_profiles']['medium']['minimum_ease_ratio'] = 0.01
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='disjoint'):
        read_config(config_path)


class NativeApiDouble:
    BW_API_ERROR_SUCCESS = 0

    def __init__(self, missing_export=False, fail_dress=False):
        self.calls = []
        self.missing_export = missing_export
        self.fail_dress = fail_dress
        self.error = 0

    def GetLastError(self):
        return self.error

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            self.error = 0
            if name == 'HostApplicationGet': return 'SYNTHETIC TEST DOUBLE; NO VSTITCHER RUN'
            if name == 'GarmentId': return 'test-garment'
            if name == 'SizeNameGet': return 'test-tight'
            if name == 'AvatarImport': return 'imported-mhr-test'
            if name == 'AvatarCurrentGet': return 'imported-mhr-test'
            if name == 'SizeCurrentGet': return 10
            if name == 'GarmentDressEx' and self.fail_dress: self.error = 99
            if name == 'GarmentSaveAs': Path(args[1]).write_bytes(b'SYNTHETIC SAVED GARMENT')
            if name == 'RenderExport3DObject' and not self.missing_export:
                export = json.loads(args[1])
                Path(export['path']).write_bytes(trimesh.creation.box().export(file_type='glb'))
        return call


def native_job(library, tmp_path):
    directory, _, _ = body(tmp_path)
    result = ClothingStage(library[0], output=tmp_path / 'prepared')(handoff_from_directory(directory))
    plan_path = Path(result['plan'])
    plan = json.loads(plan_path.read_text())
    job_path = plan_path.parent / plan['jobs'][0]['file']
    job = json.loads(job_path.read_text())
    job['output_directory'] = str(tmp_path / 'native-attempt')
    job_path.write_text(json.dumps(job))
    return job_path


def test_native_bridge_uses_imported_body_sync_simulation_and_exports(library, tmp_path):
    job_path = native_job(library, tmp_path)
    api = NativeApiDouble()
    result = run_job(job_path, api)
    names = [name for name, _ in api.calls]
    assert names.index('AvatarImport') < names.index('GarmentPrepare') < names.index('GarmentDressEx') < names.index('GarmentFinish') < names.index('RenderExport3DObject')
    simulation = json.loads(next(args[1] for name, args in api.calls if name == 'GarmentDressEx'))
    assert simulation['operation_mode'] == 'sync' and simulation['use_cache'] is False
    assert result['status'] == 'simulation_completed_unreviewed' and not result['physical_fit_verified']
    assert 'test double' in result['host_application'].lower()


@pytest.mark.parametrize('missing_export,fail_dress', [(True, False), (False, True)])
def test_native_bridge_does_not_accept_missing_export_or_api_error(library, tmp_path, missing_export, fail_dress):
    job_path = native_job(library, tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        run_job(job_path, NativeApiDouble(missing_export, fail_dress))
    assert not (tmp_path / 'native-attempt/native-result.json').exists()
