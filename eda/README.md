# Independent EDA

An independent exploratory analysis of every ECG source in the project, built from the
raw files, and a review of the earlier data pipeline against them. It is kept separate
from `ecg_experiment/` and `scripts/`, and nothing here writes to `data/` or `outputs/`.

## Notebooks

Read them in order; each builds on what the previous one established.

| Notebook | Source | Main questions |
| --- | --- | --- |
| `notebooks/01-ptbxl.ipynb` | PTB-XL | Labels, splits, shortcuts, signal integrity, processed arrays |
| `notebooks/02-mimic.ipynb` | MIMIC-IV-ECG (200k local) | Selection, lead order, machine measurements, devices |
| `notebooks/03-challenge.ipynb` | Georgia, CPSC 2018, CPSC-Extra, Chapman | Format, duplicates, clipping, union curation |
| `notebooks/04-code15.ipynb` | CODE-15% (all 18 parts) | Export archive, duration, amplitude unit, sampling rate |
| `notebooks/05-cross-dataset.ipynb` | All | Source fingerprints, pooling, scorecard and next steps |

The MIMIC notebook shows aggregate statistics only (credentialed data), so its outputs
can be kept in Git.

## Running

```bash
.venv/bin/python -m eda.compute        # build all caches once: about 80 min (MIMIC, then CODE-15 streaming)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace eda/notebooks/01-ptbxl.ipynb
```

Everything runs on CPU. Caches and figures go to `eda/outputs/`, which is gitignored.
Delete `eda/outputs/features/` to recompute from the raw files.

## Code

| Module | Purpose |
| --- | --- |
| `signals.py` | Dataset-independent features: flat leads, amplitude, limb-lead identities, noise bands, heart rate; parallel caching; ECG plots |
| `ptbxl.py`, `ptbxl_labels.py`, `ptbxl_signals.py` | PTB-XL loading, an independent re-implementation of the project label, label and signal audits |
| `mimic.py` | MIMIC record list, machine measurements, name-based lead ordering |
| `challenge.py` | Challenge headers, SNOMED names, signal hashes |
| `code15.py` | CODE-15 metadata; streams all 18 parts from the export archive one at a time |
| `processed.py` | Comparisons of the project's processed arrays with the raw files |
| `cross.py` | All sources in one table; source fingerprinting |
| `compute.py` | Builds every cache |
