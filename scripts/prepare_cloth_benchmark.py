"""Prepare real ANSUR II benchmark cases. This does not run a cloth solver."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.dataset import load_subjects

FITS = ('tight', 'medium', 'loose')
PILOT_IDS = ('ansur2_male_10027', 'ansur2_female_10037')


def metrics(subject):
    m = subject['measurements_cm']
    return {
        'height_cm': m['stature'],
        'chest_cm': m['chestcircumference'],
        'waist_cm': m['waistcircumference'],
        'hips_cm': m['buttockcircumference'],
        'waist_to_height': m['waistcircumference'] / m['stature'],
        'chest_to_hips': m['chestcircumference'] / m['buttockcircumference'],
    }


def prepare(root):
    subjects = load_subjects(root / 'data/raw/ansur2')
    by_id = {s['subject_id']: s for s in subjects}
    selected = []
    used = set()

    def add(subject, reason, stage):
        used.add(subject['subject_id'])
        selected.append({
            'subject_id': subject['subject_id'], 'sex': subject['sex'],
            'source_file': subject['source_file'], 'source_line': subject['source_line'],
            'selection_reason': reason, 'stage': stage,
            'source_measurements': metrics(subject),
            'body_status': 'existing_example' if stage == 'pilot' else 'not_generated_for_this_benchmark',
        })

    for identifier in PILOT_IDS:
        add(by_id[identifier], 'Existing fitted MHR example', 'pilot')

    for sex in ('male', 'female'):
        pool = [s for s in subjects if s['sex'] == sex]
        for metric in ('height_cm', 'waist_to_height', 'chest_to_hips'):
            ordered = sorted(pool, key=lambda s: (metrics(s)[metric], s['subject_id']))
            for quantile in (0.05, 0.95):
                target = metrics(ordered[round((len(ordered) - 1) * quantile)])[metric]
                nearest = min((s for s in ordered if s['subject_id'] not in used),
                              key=lambda s: (abs(metrics(s)[metric] - target), s['subject_id']))
                add(nearest, f'{sex}: near P{int(quantile * 100)} of {metric}', 'coverage')

    jobs = []

    def job(subject, fit, outfit, material, stage):
        key = f"{subject['subject_id']}:{fit}:{outfit}:{material}"
        jobs.append({
            'case_id': key, 'subject_id': subject['subject_id'], 'sex': subject['sex'],
            'fit': fit, 'outfit_profile': outfit, 'fabric_profile': material,
            'stage': stage, 'garments': ['shirt', 'pants'],
            'selection_seed': int(hashlib.sha256(subject['subject_id'].encode()).hexdigest()[:8], 16),
            'status': 'not_run',
        })

    for subject in selected:
        for fit in FITS:
            job(subject, fit, 'basic_shirt_pants', 'baseline', subject['stage'])
    for subject in selected[:2]:
        for fit in FITS:
            for material in ('low_bending', 'high_bending'):
                job(subject, fit, 'basic_shirt_pants', material, 'materials')
            job(subject, fit, 'alternate_shirt_pants', 'baseline', 'styles')

    assert len(selected) == len(used) == 14
    assert len(jobs) == len({j['case_id'] for j in jobs}) == 60
    assert all(j['status'] == 'not_run' for j in jobs)
    return {
        'schema_version': 1,
        'purpose': 'Case preparation only; no solver execution or quality pass is claimed',
        'source_commit': subjects[0]['source_commit'],
        'units': {'source_lengths': 'centimetres', 'ratios': 'dimensionless', 'simulation': 'metres'},
        'commercial_use_required': True,
        'backends_to_benchmark': ['newton_style3d', 'newton_vbd'],
        'backend_status': 'neither integrated or benchmarked on MHR',
        'profiles_status': 'names reserve benchmark slots; assets, dimensions and material values must be resolved before running',
        'bodies': selected,
        'cases': jobs,
        'case_counts_per_backend': {'pilot': 6, 'coverage': 36, 'materials': 12, 'styles': 6, 'total': 60},
        'timing_repetitions': {'pilot_warm_runs_per_case': 3, 'cold_run_reported_separately': True},
        'failure_tests': [
            'small_recoverable_initial_overlap', 'incorrect_leg_or_layer_order',
            'unsupported_body_or_impossible_pattern', 'missing_units_or_bad_topology',
            'worker_timeout_or_crash', 'missing_or_corrupt_garment_export',
            'resume_after_partial_fit_completion', 'configuration_or_asset_change',
            'downstream_delivery_failure',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/cloth-benchmark-plan.json')
    args = parser.parse_args()
    plan = prepare(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    print(f"Prepared {len(plan['bodies'])} subjects and {len(plan['cases'])} NOT-RUN cases per backend: {args.output}")
