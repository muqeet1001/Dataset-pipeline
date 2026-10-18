from pathlib import Path
from pipeline.dataset import load_subjects


def test_full_dataset_and_unit_conversion():
    root = Path(__file__).resolve().parents[1] / 'data/raw/ansur2'
    subjects = load_subjects(root)
    assert len(subjects) == 6068
    assert len({subject['subject_id'] for subject in subjects}) == 6068
    assert sum(subject['sex'] == 'male' for subject in subjects) == 4082
    assert sum(subject['sex'] == 'female' for subject in subjects) == 1986
    first = subjects[0]
    assert first['subject_id'] == 'ansur2_male_10027'
    assert first['measurements_cm']['stature'] == 177.6
    assert first['measurements_cm']['chestcircumference'] == 107.4
    assert first['mass_kg'] == 81.5
    assert first['source_line'] == 2
