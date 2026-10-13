# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```bash
python -m pytest pycando/tests/
# or
python -m unittest discover -s pycando/tests/
```

**Run a single test file:**
```bash
python -m pytest pycando/tests/test_builtin.py
python -m pytest pycando/tests/test_nppd.py
```

**Type-check:**
```bash
python -m mypy pycando/
```

The package has no `setup.py`/`pyproject.toml` — it is used via direct path imports in development.

## Architecture

PyCanDo is a personal Python utility library for data analysis and research work. Key modules:

- **`ak/`** — Core numerical/pandas utilities: `nppd.py` has `which`, `probability`, `center`, `data_frame`, `print_df`; `find.py` for list operations; `math.py` and `random.py` for misc math/hashing.
- **`builtin/`** — Extended Python builtins: `builtin.py` (flatten, paste, align, coalesce), `timer.py` (Times/DTimes/Timer classes), `printing.py` (indented printing), `slicing.py`, `str.py`, `files.py`.
- **`df/`** — Pandas DataFrame utilities: `condapply.py` for conditional apply, `summary.py`, `helpers.py`.
- **`stats/`** — Statistical tools: `conf_int.py` (CI, CI_CLT, CI_binom, CI_boot), `binary_confusion.py` (confusion matrix), `metrics_containers.py` (TrainTestMetric, Metrics).
- **`transformations/`** — Data transformations compatible with sklearn pipelines: `functions.py` (log1, split_pos_neg_0), `transformers.py`.
- **`plots/`** — Visualization: `rocs.py` for ROC/AUROC; datetime, timeseries, numeric, factor, and covariate plot modules.
- **`time_data/`** — Time series: `arima.py`, `date_time.py`, `holidays.py`, `windows.py`.
- **`sklearn_extensions/`** — Custom sklearn classes (e.g., `StratifiedGroupShuffleSplit`).
- **`mytypes.py`** — Central type aliases used across the package: `Int`, `Float`, `Number`, `Numeric`, `Factor`, `Variable`, `Values`, `Index`, `PositionalIndex`, etc.
- **`config/`** — Project-specific config helpers (Empirica project); not general-purpose.
- **`copies/`** — Third-party code incorporated verbatim; do not refactor.

## Code Conventions

**Module docstring header** — every file starts with a YAML-style block:
```python
"""
---
title: Module Title
version: 1.0
type: module
keywords: [keyword, ...]
description: |
    ...
file:
    date: YYYY-MM-DD
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email: [arkadiusz.kasprzyk@quantup.pl, akasp666@google.com]
"""
```

**Type annotations** — use custom aliases from `mytypes.py` rather than plain `int`/`float`/`list` where relevant. Union syntax uses `|` (Python 3.10+ style).

**Cell markers** — files use `# %%` for Spyder/notebook-style interactive sections; preserve these when editing.

**Function aliases** — many functions have short aliases (e.g., `find_all_common = find_all_0`); add aliases when a function name is long.
