# RSNA Knee Abnormality Detection — Phase 0

A deliberately small, end-to-end pipeline whose only job is to prove that a valid
`submission.csv` comes out of an offline Kaggle notebook. See [PLAN.md](PLAN.md) for the
full roadmap and [PHASE0.md](PHASE0.md) for what is and is not in scope here.

```
config.py      12 target columns + the preprocess/run config that travels with the weights
preprocess.py  preprocess_study() — the ONE code path shared by train and test
textlabel.py   precision-first multilingual pseudo-labeler (Effusion, Fracture only)
model.py       EfficientNet-B0 over slices → gated attention pool → 12 sigmoid heads
data.py        table loading + the torch Dataset (used by train.py AND the notebook)
train.py       gold ∪ pseudo-labels, masked BCE, 1 fold, gold-only validation
tools/make_fixture.py       synthetic DICOM fixture in the real schema
notebooks/infer_notebook.ipynb   offline submission notebook
tests/         pytest suite; runs entirely on the fixture, no network, no Kaggle data
```

## Quickstart (no Kaggle data needed)

```bash
pip install -r requirements.txt
pytest                                            # 74 tests, ~15s on CPU
python tools/make_fixture.py --out fixtures/synthetic
python train.py --data-dir fixtures/synthetic --out-dir artifacts \
                --epochs 2 --n-slices 8 --size 96 --no-pretrained
```

`artifacts/` then holds `model.pt`, `run_config.json` (the exact preprocessing settings)
and `metrics.json` (per-column gold AUC — never pseudo-label AUC).

To run the submission notebook locally against the fixture, point the environment
variables at it: `KNEE_CODE_DIR` (repo root), `KNEE_WEIGHTS_DIR` (`artifacts/`),
`KNEE_DATA_DIR` (fixture root), `KNEE_SPLIT=test`, `KNEE_OUT`. That is exactly what
`tests/test_notebook.py` does — the notebook that runs on Kaggle is the notebook the tests
execute.

## Running it on Kaggle (upload → attach → run offline → submit)

Train wherever you have internet (`python train.py --data-dir <competition data>` with
`efficientnet_b0` ImageNet weights), then build one directory containing `code/` (a copy of
this repo's `.py` files) and `artifacts/` (`model.pt` + `run_config.json`), and upload it as
a **Kaggle Dataset** — Datasets → New Dataset → upload the folder, e.g. named
`knee-phase0`. Open the submission notebook, add both the competition dataset and
`knee-phase0` via **Add Input**, turn **Internet OFF** in the notebook settings (the model
is built with `pretrained=False` so nothing is ever downloaded at inference), and confirm
the four path constants in the first cell match where Kaggle mounted your dataset —
defaults are `/kaggle/input/knee-phase0/code`, `/kaggle/input/knee-phase0/artifacts` and
`/kaggle/input/<competition-slug>`; override them by setting `KNEE_CODE_DIR`,
`KNEE_WEIGHTS_DIR` and `KNEE_DATA_DIR` if not. Then **Save Version → Save & Run All
(Commit)**, check the printed `TOTAL WALL CLOCK` against the 9-hour budget, and submit the
resulting `submission.csv` from the notebook's Output tab.

## Before the first real submission

`config.TARGET_COLUMNS` contains **placeholder** column names. The notebook already takes
its header from the competition's `sample_submission.csv` (authoritative), but training
does not — replace the constants in `config.py` with the real 12 names, in the real order,
as soon as you have the file.
