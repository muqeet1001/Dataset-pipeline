# ANSUR II (2012 US Army Anthropometric Survey) — public release

Real per-subject anthropometry, used so that a generated MetaHuman's 30 body constraints
come from **one measured human** rather than from independently sampled numbers. Sampling
height, weight and BMI independently can land inside every individual range and still
describe a body no human has: the covariance between girths is what makes a body read as
real, and only a per-subject record carries it.

## Provenance

| Field | Value |
| --- | --- |
| Survey | 2012 US Army Anthropometric Survey (ANSUR II) |
| Executed by | US Army Natick Soldier Research, Development and Engineering Center (NSRDEC) |
| Collected | October 2010 – April 2012 |
| Published internally | 2012 |
| Released publicly | 2017 |
| Population | Total US Army force — Active Duty, Reserves, National Guard |
| Measurements | 93 body measurements per subject, plus 3D surface scans (scans not included here) |
| Licence | US Government work, public release |

Announcement and dataset pages:

- <https://www.army.mil/article/188601/for_good_measure_natick_releases_raw_data_from_army_wide_anthropometric_survey>
- <https://www.openlab.psu.edu/datasets/ansur2/>

## Files as vendored here

| File | Rows | Columns | sha256 |
| --- | --- | --- | --- |
| `ansur2_male.csv` | 4082 | 108 | `0547aea0170e5293de519389981135e75803f02e6d40c77ace740decc0caac64` |
| `ansur2_female.csv` | 1986 | 108 | `ed7e800aa97a4d42f8286b988be2fe1883a2e789588169db959864b707d1757a` |

Row counts match the published release (4,082 male / 1,986 female), which is the check
that these are the real files and not a subset.

108 columns = 93 measurements + `subjectid` + 14 demographic/metadata fields
(`Gender`, `Date`, `Installation`, `Component`, `Branch`, `PrimaryMOS`,
`SubjectsBirthLocation`, `SubjectNumericRace`, `Ethnicity`, `DODRace`, `Age`, `Heightin`,
`Weightlbs`, `WritingPreference`).

## Reading these files — three traps, all measured

1. **Encoding is not UTF-8.** Both files contain Latin-1 bytes (a `0xe9` at male byte
   offset 3008). `open(..., encoding="utf-8")` raises. Read as `latin-1`.
2. **Linear measurements are millimetres**, not centimetres. `stature` male mean is
   1756.2. Divide by 10 for cm.
3. **`weightkg` is in units of 0.1 kg.** Male mean 855.2 → 85.5 kg; female mean 677.6 →
   67.8 kg. Divide by 10.
4. The male file's id column is `subjectid`; the female file's is `SubjectId`. Callers
   must lower-case the header row. `Dataset.core.anthropometry` does.

## Demographic scope — state this wherever the data is used

This is a **military** population: fitter and younger than the civilian public, and
selected against the extremes of body mass by service entry standards. It is not a
general-population sample. It is used here because it is the largest public dataset with
**per-subject** whole-body measurement vectors, which is the property that matters for
generating coherent bodies. Any claim about civilian representativeness would need a
different source (e.g. NHANES for height/weight/circumference distributions, which
however does not publish full per-subject measurement vectors).
