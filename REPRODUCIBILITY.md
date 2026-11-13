# Reproducibility status (pre-publication review)

The following checks were performed **in an existing environment** on 2026-11-13, not a newly built clean virtual environment:

```bash
python -m compileall -q delta_core.py adapters.py stream_cli.py experiments.py create_samples.py v36_original.py
python experiments.py --quick
python experiments.py
```

Both experiment commands completed with exit code 0. Comparing `results/results.json` with the existing research reference showed matching structured values **except** recorded wall-clock times (`long_run[*].seconds` and `elapsed_seconds`). This is expected: timings depend on execution environment. Full results can be regenerated locally and may overwrite files in `results/`.

Environment used for release review:

- Python 3.13.5
- numpy 2.3.5
- scipy 1.17.0
- matplotlib 3.10.8
- scikit-learn 1.8.0
- Pillow 12.3.0

`requirements.txt` contains open version ranges; exact reproducibility across platforms or fresh package installs is **not yet established**. Tests do not establish performance outside their specified synthetic or small benchmark settings.