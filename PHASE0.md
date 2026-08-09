# Phase 0 — valid offline submission

**Goal: prove the pipeline, not win.** With 58 gold labels, do not expect a leaderboard
position.

## Context (from PLAN.md)

- Only 58 of 4,407 studies have gold labels. Text is a SUPERVISION channel, not a model
  input. The test set has NO reports.
- Metric: macro-AUC over 12 findings. Submission: `submission.csv`, 12 probability columns.
- Submission notebook: internet OFF, ≤ 9 h. Decode is cheap (~0.28 s/series).

## Non-negotiable design invariants

1. ONE `preprocess_study(study_uid, series_df, plane_prefs, n_slices, size, max_series)`
   used identically for train and test. It must RETURN the stacked multi-series form even
   when called with `max_series=1`. No test-only code path.
2. Masked BCE from the start: loss backprops only on labeled columns.
3. Validate on gold only. Never report pseudo-label AUC as if it were truth.

## Build order (as built)

| # | Deliverable | File |
|---|---|---|
| 1 | Synthetic fixture: ~12 fake studies, real DICOMs, real schema, mixed planes/languages | `tools/make_fixture.py` |
| 2 | `preprocess_study()`: position-sorted slices, per-series 1–99 percentile clip, resize, uint8; canonical series = sagittal fluid-sensitive with sagittal-any fallback | `preprocess.py` |
| 3 | Precision-first pseudo-labeler for Effusion + Fracture only, en/es/tr/de/nl, with a per-(study, column) mask | `textlabel.py` |
| 4 | EfficientNet-B0 shared over slices → gated attention pool → 12 sigmoid heads, input `(B, n_slices, 3, H, W)` | `model.py` |
| 5 | Train on gold ∪ high-precision pseudo-labels, masked BCE, 1 fold; save weights + exact preprocess config | `train.py` |
| 6 | OFFLINE inference notebook writing a schema-valid `submission.csv`, prints wall clock | `notebooks/infer_notebook.ipynb` |

Shared table loading and the torch `Dataset` live in `data.py`; the 12 column names and the
preprocess/run config live in `config.py`.

## Definition of done

- `pytest` green on the fixture — 74 tests, no network, no Kaggle data.
- `infer_notebook.ipynb` produces a schema-valid `submission.csv` on the fixture
  (`tests/test_notebook.py` executes the actual notebook).
- README explains the Kaggle Dataset → attach → run offline → submit loop.

## Explicitly OUT of scope for Phase 0

Multi-series fusion, orientation/laterality canonicalization, the other 10 text columns,
soft labels, larger backbones. TODO hooks are left in place (`grep -rn "TODO(phase"`), and
nothing beyond them was built.

## Known Phase-0 limitations (deliberate, not bugs)

- `config.TARGET_COLUMNS` holds **placeholder** column names. The notebook reads the real
  header from the competition's `sample_submission.csv` when present; training does not.
  Swap the constants once the real file is in hand.
- The pseudo-labeler is regex+negation, which §2.3 of PLAN.md shows is unsafe as a general
  solution. It is only safe here because it abstains on anything hedged and labels two
  columns. It is not the Phase-1 answer.
- Absence of any mention is NaN, never 0.
- `to_model_input` replicates the grey channel three times rather than stacking adjacent
  slices (2.5D).
- Only the first series slot is fed to the model, even though `preprocess_study` returns
  all of them.
