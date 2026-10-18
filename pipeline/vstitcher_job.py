"""Run inside licensed VStitcher-CLI; imports only stdlib and its native BwApi."""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def run_job(job_path, api):
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    if job.get('schema_version') != 1:
        raise ValueError('Unsupported VStitcher job schema')
    for path_key, hash_key in (('avatar_path', 'avatar_sha256'), ('garment_path', 'garment_sha256')):
        if digest(job[path_key]) != job[hash_key]:
            raise ValueError(f'Input changed: {path_key}')
    output = Path(job['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('dressed.bw', 'scene.glb', 'garment.glb')):
        raise ValueError('Use a fresh native attempt directory; existing exports are preserved')
    calls = []

    def checked(name, *arguments):
        value = getattr(api, name)(*arguments)
        code = api.GetLastError()
        if code != api.BW_API_ERROR_SUCCESS:
            raise RuntimeError(f'BwApi.{name} failed: {code}')
        calls.append(name)
        return value

    host = checked('HostApplicationGet')
    checked('GarmentOpen', job['garment_path'])
    garment_id = checked('GarmentId')
    size_id = job['selection']['native_size_id']
    if checked('SizeNameGet', garment_id, size_id) != job['selection']['size_name']:
        raise ValueError('Native size ID/name differs from the enrolled garment catalog')
    checked('SizeCurrentSet', garment_id, size_id)
    reference = job.get('avatar_reference_vsa')
    if reference:
        ids = checked('AvatarImportVSA', reference)
        if len(ids) != 1:
            raise ValueError('Avatar reference VSA must contain one compatible reference avatar')
        avatar_id = checked('AvatarImportEx', json.dumps({'path': job['avatar_path'], 'reference': ids[0]}))
    else:
        avatar_id = checked('AvatarImport', job['avatar_path'])
    if not avatar_id:
        raise ValueError('VStitcher did not return an imported MHR avatar ID')
    checked('AvatarCurrentSet', avatar_id)
    checked('GarmentPrepare', garment_id)
    # Documented synchronous mode prevents exporting before the dress iterations finish.
    checked('GarmentDressEx', garment_id, json.dumps({
        'operation_mode': 'sync', 'iterations': job['simulation_iterations'],
        'use_cache': False, 'symmetric_simulation': False,
    }))
    checked('GarmentFinish', garment_id)
    if checked('AvatarCurrentGet') != avatar_id or checked('SizeCurrentGet', garment_id) != size_id:
        raise ValueError('VStitcher changed the requested MHR avatar or garment size')
    checked('GarmentSaveAs', garment_id, str(output / 'dressed.bw'))
    for name, include_avatar in (('scene.glb', True), ('garment.glb', False)):
        export = {
            '3d_format': 'gltf', 'path': str(output / name), 'type': 'binary',
            'include_avatar': include_avatar, 'match_garment': False,
            'up_axis': 'y', 'scale': 1.0, 'use_pattern_pieces_names': True,
            'layout': {'layout_type': 'layout_uv', 'piece': 'all_pieces_bounding_box', 'dpi': 150, 'file_format': 'png'},
            'alpha_mode': 'opaque',
        }
        checked('RenderExport3DObject', garment_id, json.dumps(export))
    files = ('dressed.bw', 'scene.glb', 'garment.glb')
    if any(not (output / name).is_file() or (output / name).stat().st_size == 0 for name in files):
        raise ValueError('VStitcher returned without all nonempty garment and scene exports')
    if digest(job['garment_path']) != job['garment_sha256']:
        raise ValueError('Native source garment copy was modified in place')
    result = {
        'schema_version': 1, 'job_id': job['job_id'], 'subject_id': job['subject_id'],
        'status': 'simulation_completed_unreviewed', 'output_directory': str(output),
        'host_application': host, 'imported_avatar_id': avatar_id, 'native_size_id': size_id,
        'fit_condition_target': job['selection']['fit'], 'physical_fit_verified': False,
        'simulation_iterations': job['simulation_iterations'], 'operation_mode': 'sync',
        'source_inputs': {'avatar_sha256': job['avatar_sha256'], 'garment_sha256': job['garment_sha256'], 'source_mesh_sha256': job['source_mesh_sha256']},
        'sha256': {name: digest(output / name) for name in files}, 'api_calls': calls,
    }
    with (output / 'native-result.json').open('w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    checked('GarmentCloseEx', garment_id, json.dumps({'discard_changes': True}))
    return result


if os.environ.get('DATASET_PIPELINE_VSTITCHER_JOB'):
    import BwApi
    run_job(os.environ['DATASET_PIPELINE_VSTITCHER_JOB'], BwApi)
