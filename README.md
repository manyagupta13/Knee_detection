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
notebooks/train_notebook.ipynb   Kaggle training notebook — internet ON, clones this repo
notebooks/infer_notebook.ipynb   Kaggle submission notebook — internet OFF
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

## Running it on Kaggle (clone → train → attach → run offline → submit)

Two notebooks, two Kaggle notebook sessions. No manual zipping/uploading of code — the
training notebook clones this repo directly.

**1. Training notebook — internet ON.**

- On Kaggle: **Create → New Notebook**, then **File → Import Notebook** and upload
  `notebooks/train_notebook.ipynb` from this repo (or paste its cells into a new notebook).
- **Add Input** the competition dataset.
- In notebook **Settings**, confirm **Internet: On**. GPU accelerator is optional but
  recommended.
- The repo is public, so the first cell's `git clone` needs no credentials. If you're
  working from a fork, edit `REPO_URL`/`REPO_BRANCH` in that cell. Once this branch is
  merged, change `REPO_BRANCH` to `"main"`.
- Check `DATA_DIR` in the first cell against whatever Kaggle actually mounted
  (`ls /kaggle/input`) and fix it if the competition slug differs.
- **Save Version → Save & Run All (Commit).** This clones the repo into
  `/kaggle/working/code`, installs `pydicom`/`timm`, reads the real 12 column names from
  the competition's `sample_submission.csv`, trains, and writes `code/` + `artifacts/`
  (`model.pt`, `run_config.json`, `metrics.json`) into `/kaggle/working` — which becomes
  this notebook's **Output**, a Kaggle Dataset you can attach elsewhere (e.g.
  `your-username/knee-phase0-train`).

**2. Submission notebook — internet OFF.**

- Import `notebooks/infer_notebook.ipynb` the same way.
- **Add Input** both the competition dataset and the training notebook's output dataset
  from step 1.
- In notebook **Settings**, turn **Internet: Off** (the model is built with
  `pretrained=False` here, so nothing is ever downloaded).
- The first cell reads `KNEE_CODE_DIR`/`KNEE_WEIGHTS_DIR` — defaults are
  `/kaggle/input/knee-phase0/code` and `/kaggle/input/knee-phase0/artifacts`; if your output
  dataset mounted under a different slug (Kaggle names it after the source notebook by
  default), set those two Kaggle notebook **environment variables** in Settings, or just
  edit the cell to match.
- **Save Version → Save & Run All (Commit)**, check the printed `TOTAL WALL CLOCK` against
  the 9-hour budget, and submit the resulting `submission.csv` from the notebook's Output
  tab.

## Before the first real submission

`config.TARGET_COLUMNS` contains **placeholder** column names. The notebook already takes
its header from the competition's `sample_submission.csv` (authoritative), but training
does not — replace the constants in `config.py` with the real 12 names, in the real order,
as soon as you have the file.
