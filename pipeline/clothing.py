"""Prepare and optionally execute three sex-matched VStitcher garment jobs."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import uuid
import numpy as np
import trimesh
from .runner import fingerprint, write_json

FIT_NAMES = ('tight', 'medium', 'loose')
MEASUREMENTS = ('chestcircumference', 'waistcircumference', 'buttockcircumference')


def digest(path: Path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_config(path: Path):
    path = path.resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1 or type(config.get('seed')) is not int:
        raise ValueError('Clothing config requires schema_version 1 and an integer seed')
    previous_max = -1.0
    settings = config['vstitcher']
    if type(settings['simulation_iterations']) is not int or settings['simulation_iterations'] < 1:
        raise ValueError('VStitcher simulation_iterations must be a positive integer')
    if not isinstance(settings['timeout_seconds'], (int, float)) or not math.isfinite(settings['timeout_seconds']) or settings['timeout_seconds'] <= 0:
        raise ValueError('VStitcher timeout_seconds must be positive and finite')
    if settings.get('avatar_reference_vsa'):
        settings['avatar_reference_vsa'] = str((path.parent / settings['avatar_reference_vsa']).resolve())
    for name in FIT_NAMES:
        band = config['fit_profiles'][name]
        low, high = band['minimum_ease_ratio'], band['maximum_ease_ratio']
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (low, high)) or not 0 <= low < high <= 2 or low < previous_max:
            raise ValueError('Fit ease bands must be finite, ordered, nonnegative and disjoint')
        previous_max = high
    catalog_path = (path.parent / config['catalog']).resolve()
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if catalog.get('schema_version') != 1 or not isinstance(catalog.get('garments'), list):
        raise ValueError('Garment catalog requires schema_version 1 and garments list')
    identifiers = set()
    for garment in catalog['garments']:
        if not isinstance(garment['id'], str) or not garment['id'] or garment['id'] in identifiers:
            raise ValueError('Garment IDs must be unique nonempty strings')
        identifiers.add(garment['id'])
        if garment['sex'] not in ('male', 'female') or garment['measurement'] not in MEASUREMENTS:
            raise ValueError('Each garment requires male/female sex and a supported circumference measurement')
        if len(garment['sha256']) != 64 or any(character not in '0123456789abcdef' for character in garment['sha256']):
            raise ValueError('Native garment SHA-256 is required')
        if Path(garment['file']).suffix.lower() not in ('.bw', '.vsgx'):
            raise ValueError('Catalog garments must be native .bw or .vsgx files')
        if not garment.get('measurement_source'):
            raise ValueError('Record how the native garment circumferences were measured')
        sizes = set()
        for size in garment['sizes']:
            if type(size['native_size_id']) is not int or size['native_size_id'] < 0 or size['native_size_id'] in sizes:
                raise ValueError('Native size IDs must be unique integers within a garment')
            sizes.add(size['native_size_id'])
            circumference = size['circumference_cm']
            if not size['name'] or not isinstance(circumference, (int, float)) or not math.isfinite(circumference) or circumference <= 0:
                raise ValueError('Every size needs its actual name and positive measured circumference')
    return config, catalog, catalog_path


def select_sizes(subject: dict, fit_report: dict, config: dict, catalog: dict):
    """Select one random compatible garment and one distinct graded size per fit."""
    seed = fingerprint({'seed': config['seed'], 'subject_id': subject['subject_id']})
    rng = random.Random(int(seed, 16))
    eligible = []
    matching = [garment for garment in catalog['garments'] if garment['sex'] == subject['sex']]
    for garment in sorted(matching, key=lambda item: item['id']):
        body_cm = fit_report['measurements'][garment['measurement']]['achieved_cm']
        if not isinstance(body_cm, (int, float)) or not math.isfinite(body_cm) or body_cm <= 0:
            raise ValueError('Accepted body circumference must be finite and positive')
        choices = {}
        for name in FIT_NAMES:
            band = config['fit_profiles'][name]
            choices[name] = [
                size for size in sorted(garment['sizes'], key=lambda item: item['native_size_id'])
                if band['minimum_ease_ratio'] <= (size['circumference_cm'] - body_cm) / body_cm < band['maximum_ease_ratio']
            ]
        if all(choices.values()):
            eligible.append((garment, choices, body_cm))
    if not eligible:
        reason = 'missing_sex_matched_garments' if not matching else 'no_garment_with_three_matching_sizes'
        return None, reason
    garment, choices, body_cm = rng.choice(eligible)
    selected = []
    for name in FIT_NAMES:
        size = rng.choice(choices[name])
        selected.append({
            'fit': name, 'native_size_id': size['native_size_id'], 'size_name': size['name'],
            'garment_circumference_cm': size['circumference_cm'], 'body_circumference_cm': body_cm,
            'ease_cm': size['circumference_cm'] - body_cm,
            'ease_ratio': (size['circumference_cm'] - body_cm) / body_cm,
            'measurement': garment['measurement'],
        })
    return {'garment': garment, 'variants': selected}, None


def handoff_from_directory(directory: Path):
    directory = directory.resolve()
    names = ('body.ply', 'mhr-inputs.npz', 'source-record.json', 'fit-report.json')
    subject = json.loads((directory / 'source-record.json').read_text(encoding='utf-8'))
    return {
        'subject_id': subject['subject_id'], 'output_root': str(directory),
        'files': {name: name for name in names}, 'sha256': {name: digest(directory / name) for name in names},
        'idempotency_key': fingerprint({'subject_id': subject['subject_id'], 'mesh': digest(directory / 'body.ply')}),
    }


class ClothingStage:
    def __init__(self, config_path: Path, mode='prepare', output: Path | None = None):
        if mode not in ('prepare', 'simulate'):
            raise ValueError('Clothing mode must be prepare or simulate')
        self.config_path = config_path.resolve()
        self.config, self.catalog, self.catalog_path = read_config(self.config_path)
        self.mode, self.output = mode, output.resolve() if output else None
        if mode == 'simulate':
            executable = self.config['vstitcher'].get('cli_executable')
            if not executable:
                raise RuntimeError('VStitcher access is not configured; use prepare mode until the licensed CLI is available')
            if not (self.config_path.parent / executable).is_file():
                raise FileNotFoundError(f'VStitcher CLI not found: {executable}')
            if not self.catalog['garments']:
                raise ValueError('Enroll real native garments before starting VStitcher simulation')
        native_script = Path(__file__).with_name('vstitcher_job.py')
        self.config_key = fingerprint({
            'config': self.config, 'catalog': self.catalog, 'mode': mode,
            'code': digest(Path(__file__)), 'native_code': digest(native_script),
        })
        self.stage_id = 'vstitcher:' + self.config_key

    def __call__(self, handoff: dict):
        root = Path(handoff['output_root']).resolve()
        paths = {name: (root / handoff['files'][name]).resolve() for name in handoff['sha256']}
        for name, path in paths.items():
            if not path.is_relative_to(root) or digest(path) != handoff['sha256'][name]:
                raise ValueError(f'MHR handoff checksum/path mismatch: {name}')
        subject = json.loads(paths['source-record.json'].read_text(encoding='utf-8'))
        report = json.loads(paths['fit-report.json'].read_text(encoding='utf-8'))
        if subject['subject_id'] != handoff['subject_id'] or report['subject_id'] != handoff['subject_id'] or report['status'] != 'accepted_proxy_fit' or report['mesh_sha256'] != handoff['sha256']['body.ply']:
            raise ValueError('Clothing stage requires an accepted matching MHR subject')
        if subject['sex'] not in ('male', 'female'):
            raise ValueError('Subject has no supported recorded sex for garment selection')
        base = self.output or (root / 'clothing')
        directory = base / subject['subject_id'] / self.config_key[:16]
        directory.mkdir(parents=True, exist_ok=True)
        avatar = trimesh.load_mesh(paths['body.ply'], process=False)
        if not np.isfinite(avatar.vertices).all() or not avatar.is_watertight:
            raise ValueError('VStitcher avatar must be finite and closed')
        if abs(float(avatar.extents[1]) - report['height_achieved_cm']) > 5e-5 or abs(float(avatar.vertices[:, 1].min())) > 5e-5:
            raise ValueError('Avatar height or Y-up grounding differs from the accepted mesh')
        avatar_path = directory / 'avatar.obj'
        avatar_path.write_text(
            '# Dataset-pipeline: centimetres; Y up; feet at Y=0\n' + trimesh.exchange.obj.export_obj(avatar, include_normals=True, include_texture=False, digits=8),
            encoding='utf-8',
        )
        loaded = trimesh.load_mesh(avatar_path, process=False)
        if not np.array_equal(avatar.faces, loaded.faces) or not np.allclose(avatar.vertices, loaded.vertices, atol=1e-7, rtol=0):
            raise ValueError('VStitcher OBJ export changed the MHR geometry')
        selection, reason = select_sizes(subject, report, self.config, self.catalog)
        plan = {
            'schema_version': 1, 'stage': 'vstitcher_clothing', 'subject_id': subject['subject_id'],
            'sex': subject['sex'], 'seed': self.config['seed'], 'config_fingerprint': self.config_key,
            'source_mesh_sha256': handoff['sha256']['body.ply'],
            'avatar': {'file': 'avatar.obj', 'sha256': digest(avatar_path), 'units': 'cm', 'vertical_axis': 'Y', 'height_cm': float(avatar.extents[1])},
            'fit_label_kind': 'intended_from_measured_rest_ease; physical fit not yet verified',
            'requested_fits': list(FIT_NAMES), 'native_simulation_executed': False,
            'status': 'waiting_for_garment_catalog' if reason else 'prepared_not_executed',
            'reason': reason, 'jobs': [],
        }
        if selection:
            garment = selection['garment']
            source = (self.catalog_path.parent / garment['file']).resolve()
            if not source.is_file() or digest(source) != garment['sha256']:
                raise ValueError(f'Native garment file missing or checksum differs: {garment["id"]}')
            for selected in selection['variants']:
                variant = directory / selected['fit']
                variant.mkdir(exist_ok=True)
                working = variant / ('input' + source.suffix.lower())
                shutil.copyfile(source, working)
                job = {
                    'schema_version': 1, 'subject_id': subject['subject_id'],
                    'job_id': fingerprint({'handoff': handoff['idempotency_key'], 'config': self.config_key, 'selection': selected, 'garment': garment['sha256']}),
                    'avatar_path': str(avatar_path), 'avatar_sha256': plan['avatar']['sha256'],
                    'source_mesh_sha256': handoff['sha256']['body.ply'],
                    'garment_id': garment['id'], 'garment_path': str(working), 'garment_sha256': garment['sha256'],
                    'selection': selected, 'sex': subject['sex'],
                    'simulation_iterations': self.config['vstitcher']['simulation_iterations'],
                    'avatar_reference_vsa': self.config['vstitcher'].get('avatar_reference_vsa'),
                    'status': 'prepared_not_executed',
                }
                write_json(variant / 'job.json', job)
                plan['jobs'].append({'fit': selected['fit'], 'file': selected['fit'] + '/job.json', 'job_id': job['job_id'], 'selection': selected})
        write_json(directory / 'plan.json', plan)
        if self.mode == 'prepare' or reason:
            return {'status': 'pending', 'phase': 'preparation', 'plan': str(directory / 'plan.json'), 'reason': reason or 'waiting_for_vstitcher_access', 'variants_prepared': len(plan['jobs'])}
        receipts = [self.execute_job(directory / item['file']) for item in plan['jobs']]
        plan.update(status='simulation_completed_unreviewed', native_simulation_executed=True, results=receipts)
        write_json(directory / 'plan.json', plan)
        return {'status': 'completed', 'plan': str(directory / 'plan.json'), 'variants': receipts, 'physical_fit_verified': False}

    def execute_job(self, job_path: Path):
        settings = self.config['vstitcher']
        executable = settings.get('cli_executable')
        if not executable:
            raise RuntimeError('VStitcher access is not configured; use prepare mode until the licensed CLI is available')
        executable = (self.config_path.parent / executable).resolve()
        if not executable.is_file():
            raise FileNotFoundError(f'VStitcher CLI not found: {executable}')
        job = json.loads(job_path.read_text(encoding='utf-8'))
        result_path = job_path.parent / 'result.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if self.valid_result(result, job['job_id']):
                return result
        attempt = job_path.parent / ('attempt-' + uuid.uuid4().hex[:12])
        attempt.mkdir()
        job['output_directory'] = str(attempt)
        write_json(attempt / 'native-job.json', job)
        environment = os.environ.copy()
        environment['DATASET_PIPELINE_VSTITCHER_JOB'] = str(attempt / 'native-job.json')
        script = Path(__file__).with_name('vstitcher_job.py')
        with (attempt / 'stdout.log').open('w', encoding='utf-8') as stdout, (attempt / 'stderr.log').open('w', encoding='utf-8') as stderr:
            completed = subprocess.run([str(executable), '--script', str(script)], env=environment, stdout=stdout, stderr=stderr, timeout=settings['timeout_seconds'], check=False)
        if completed.returncode != 0:
            raise RuntimeError(f'VStitcher failed with code {completed.returncode}; logs: {attempt}')
        result = json.loads((attempt / 'native-result.json').read_text(encoding='utf-8'))
        if not self.valid_result(result, job['job_id']):
            raise ValueError('Native VStitcher export receipt or output checksums differ')
        for name in ('scene.glb', 'garment.glb'):
            scene = trimesh.load_scene(Path(result['output_directory']) / name)
            if not scene.geometry or not all(np.isfinite(geometry.vertices).all() and len(geometry.faces) > 0 for geometry in scene.geometry.values()):
                raise ValueError(f'VStitcher exported invalid geometry: {name}')
        write_json(result_path, result)
        return result

    @staticmethod
    def valid_result(result: dict, expected_job_id=None):
        try:
            if result.get('status') != 'simulation_completed_unreviewed' or result.get('job_id') != (expected_job_id or result['job_id']):
                return False
            root = Path(result['output_directory'])
            return set(result['sha256']) == {'scene.glb', 'garment.glb', 'dressed.bw'} and all((root / name).is_file() and digest(root / name) == checksum for name, checksum in result['sha256'].items())
        except (OSError, KeyError, TypeError):
            return False

    def validate_receipt(self, receipt):
        return receipt.get('status') == 'completed' and len(receipt.get('variants', [])) == 3 and all(self.valid_result(result) for result in receipt['variants'])
