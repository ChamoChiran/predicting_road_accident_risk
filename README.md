# Road Accident Risk

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Predicting road accident risk using structured data and reproducible ML workflows.

This repository contains code, data conventions, and tooling to explore, train, and evaluate models that predict road accident risk. It follows a lightweight reproducible project layout suitable for experimentation and small-scale model training.

Key goals:
- Provide reproducible data preparation and feature engineering.
- Offer simple training and inference entrypoints.
- Keep experiments, artifacts, and documentation organized.

---

Table of contents
- Overview
- Quickstart
- Data layout
- Project structure
- Usage examples
- Tests
- Development notes
- Contributing
- License

---

Overview
--------
This project prepares a pipeline for predicting road accident risk from tabular data. The codebase contains utilities to load and preprocess data, build features, train simple models, and run inference. Exploratory notebooks and reporting utilities live under `notebooks/` and `reports/`.

Quickstart
----------
The repository includes both a conda environment file (`environment.yml`) and a pip/requirements file. Pick one workflow below.

1) Conda (recommended):

```bash
# create and activate environment
conda env create -f environment.yml -n road-risk
conda activate road-risk
```

2) Virtualenv / pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After dependencies are installed, prepare data and run training (examples below).

Data layout
-----------
Data is organized under the `data/` directory. The repository currently includes a `data/processed/` with sample CSVs.

- data/raw/       — Original raw data (kept immutable).
- data/interim/   — Intermediate transformed datasets.
- data/processed/ — Final datasets used for modeling (train/test/sample_submission).
- data/external/  — Third-party reference datasets (if any).

Example files present (commit snapshots):
- `data/processed/train.csv` — training dataset
- `data/processed/test.csv` — test dataset
- `data/processed/sample_submission.csv` — sample submission format

Project structure
-----------------
Top-level layout (relevant files/folders):

- `road_accident_risk/` — Source package
  - `config.py`       — project paths and config values
  - `dataset.py`      — data loading / ingestion helpers
  - `features.py`     — feature engineering utilities
  - `plots.py`        — plotting helpers for EDA and reports
  - `modeling/`       — training and prediction entrypoints
    - `train.py`      — training script / logic
    - `predict.py`    — inference utilities
- `notebooks/`        — exploratory notebooks
- `models/`           — (gitignored) saved model artifacts
- `reports/`          — generated analysis and figures
- `tests/`            — unit tests (pytest)

Usage examples
--------------
The repository includes a `Makefile` for convenience. Use it if available, or call the python modules directly.

Using Makefile (if targets exist):

```bash
# prepare data (if a make target exists)
make data

# train a model (example)
make train
```

Direct module entry (recommended when experimenting):

```bash
# train (adjust flags or config as implemented in road_accident_risk.modeling.train)
python -m road_accident_risk.modeling.train --help
python -m road_accident_risk.modeling.train --config configs/train.yaml

# predict / run inference
python -m road_accident_risk.modeling.predict --model models/latest.pkl --input data/processed/test.csv --output predictions.csv
```

Note: The CLI flags and config locations depend on the implementation in `road_accident_risk/modeling/*.py`. Inspect those files for exact arguments.

Tests
-----
Run the test suite with pytest:

```bash
pytest -q
```

The `tests/` directory contains fast unit tests for data utilities and basic sanity checks. Add tests for new functionality and keep them deterministic and small.

Development notes
-----------------
Formatting and linting
- The project includes `pyproject.toml` / `setup.cfg`—use `black` and `flake8` for formatting and linting. Example:

```bash
# format
black .

# lint
flake8
```

Documentation
- A `docs/` folder exists and is configured for mkdocs; to serve docs locally:

```bash
pip install mkdocs
mkdocs serve
```

Reproducible experiments
- Keep intermediate artifacts out of version control. Use `models/` to store trained artifacts (this folder is typically gitignored).
- Use the notebooks for exploratory analysis and move stable code into the `road_accident_risk` package.

Contributing
------------
Contributions are welcome. A suggested workflow:
- Fork the repo and create a feature branch.
- Add tests for new behavior.
- Run formatting and linting locally.
- Open a pull request describing the change.

If you plan to add large dataset downloads or credentials, prefer documenting the steps in `road_accident_risk/dataset.py` and ensure secrets are never committed.

License
-------
This repository includes a `LICENSE` file at the top level. Refer to that file for license terms.

Contact
-------
If you want help improving the README or adding specific run examples (for example, exact CLI flags for `train.py`), tell me which workflow you use (conda or pip) and I will add runnable commands that match the code in `road_accident_risk/modeling/`.

---

Happy experimenting — make small changes, add tests, and iterate quickly.
