"""Single source of truth for target columns and the preprocessing contract.

Everything that both training and inference must agree on lives here, so that
the submission notebook cannot silently drift from the training run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
# The 12 findings scored by macro-AUC, in submission order. Verified against the
# competition's sample_submission.csv header — keep these byte-exact, including
# the apostrophe in "Baker's" and the spaces.
TARGET_COLUMNS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

N_TARGETS = len(TARGET_COLUMNS)

ID_COLUMN = "StudyInstanceUID"

# Columns the Phase-0 text pseudo-labeler was allowed to touch. Phase 1 widened
# this to all 12; the constant is kept for the Phase-0 regression tests.
PSEUDO_LABEL_COLUMNS: tuple[str, ...] = ("Effusion", "Fracture")

# MEASURED positive-class precision of the text labeler against the 58 gold
# studies (evaluate_labeler.py, latest run). These are not guesses - rerun
# the evaluator and update them whenever the lexicon changes.
#
# Used to WEIGHT the loss per column rather than to include/exclude columns
# outright: a column labeled at 0.67 precision across 866 studies still carries
# more signal than 41 gold studies alone, but it should not shout as loudly as
# ACL at 0.90. Gold labels always carry weight 1.0 regardless.
PSEUDO_LABEL_PRECISION: dict[str, float] = {
    "ACL": 0.94,   # with mention-implies-negative enabled for this column
    "MCL": 0.67,
    "Medial Meniscus": 0.78,
    "Lateral Meniscus": 0.89,
    "Medial OA": 0.67,
    "Lateral OA": 0.67,
    "PF OA": 0.67,
    "Effusion": 0.75,
    "Synovitis": 0.70,
    "Baker's": 0.67,
    "Contusion": 0.56,  # weakest column; three lexicon passes failed to lift it
    "Fracture": 0.875,
}

# Map precision -> loss weight. 0.5 precision is a coin flip and earns weight 0;
# perfect precision earns 1.0. Linear in between.
def confidence_weight(precision: float) -> float:
    return max(0.0, min(1.0, (precision - 0.5) * 2.0))


PSEUDO_LABEL_WEIGHTS: dict[str, float] = {
    col: confidence_weight(p) for col, p in PSEUDO_LABEL_PRECISION.items()
}


def target_columns(sample_submission: str | Path | None = None) -> list[str]:
    """Return the 12 target columns, preferring the real competition header.

    If ``sample_submission`` points at a readable CSV, its column order (minus
    the id column) wins. This is the only safe way to guarantee the submission
    header matches, since the placeholder names above may be wrong.
    """
    if sample_submission is not None:
        path = Path(sample_submission)
        if path.exists():
            header = path.read_text(encoding="utf-8").splitlines()[0]
            cols = [c.strip() for c in header.split(",")]
            cols = [c for c in cols if c != ID_COLUMN]
            if len(cols) != N_TARGETS:
                raise ValueError(
                    f"{path} has {len(cols)} non-id columns, expected {N_TARGETS}"
                )
            return cols
    return list(TARGET_COLUMNS)


# ---------------------------------------------------------------------------
# Preprocessing contract
# ---------------------------------------------------------------------------
# Plane preference order used to pick the canonical series. Each entry is
# "<plane>_<contrast>"; "any" matches anything. Resolution is first-match-wins,
# so the ordering below is: sagittal fluid-sensitive, then sagittal anything.
DEFAULT_PLANE_PREFS: tuple[str, ...] = ("sagittal_fluid", "sagittal_any")

# Multi-plane preference order. One series is taken per entry, so this yields
# sagittal + coronal + axial rather than three sagittals. Findings are
# plane-specific: MCL and the femorotibial compartments are coronal calls,
# patellofemoral cartilage is axial, the cruciates and meniscal horns sagittal.
MULTI_PLANE_PREFS: tuple[str, ...] = (
    "sagittal_fluid",
    "coronal_fluid",
    "axial_fluid",
    "sagittal_any",
    "coronal_any",
    "axial_any",
)


def balance_factor(positive_rate: float) -> float:
    """Down-weight columns whose pseudo-labels are nearly all one class.

    A column labeled 95% positive (Synovitis was 377/19) carries almost no
    contrast: the model can satisfy the loss by predicting the majority and
    still score AUC 0.5 - which is exactly what Synovitis did, coming back at
    0.424, the only column to get worse. Peaks at 1.0 for a balanced column and
    falls to 0 at either extreme.
    """
    p = min(max(float(positive_rate), 0.0), 1.0)
    return 4.0 * p * (1.0 - p)


@dataclass(frozen=True)
class PreprocessConfig:
    """Exact preprocessing settings. Saved next to the weights, reloaded at
    inference time so train and test cannot diverge."""

    n_slices: int = 16
    size: int = 224
    max_series: int = 1  # TODO(phase-2): raise for multi-series fusion
    plane_prefs: tuple[str, ...] = DEFAULT_PLANE_PREFS
    clip_percentiles: tuple[float, float] = (1.0, 99.0)
    # Mirror left knees into the right-knee frame so the model learns each
    # finding once instead of twice. Changes the pixels, so it MUST be part of
    # the cache key - hence it lives here, not as a loose function argument.
    canonicalize: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["plane_prefs"] = list(self.plane_prefs)
        d["clip_percentiles"] = list(self.clip_percentiles)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(
            n_slices=int(d["n_slices"]),
            size=int(d["size"]),
            max_series=int(d["max_series"]),
            plane_prefs=tuple(d["plane_prefs"]),
            clip_percentiles=tuple(float(x) for x in d["clip_percentiles"]),
            canonicalize=bool(d.get("canonicalize", True)),
        )


@dataclass
class RunConfig:
    """Everything the inference notebook needs to rebuild the model."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    backbone: str = "efficientnet_b0"
    target_columns: Sequence[str] = TARGET_COLUMNS
    # "2.5d" stacks adjacent slices as channels; "grey" replicates one slice.
    # Saved with the weights so inference cannot silently use the other one.
    input_mode: str = "2.5d"

    def to_dict(self) -> dict:
        return {
            "preprocess": self.preprocess.to_dict(),
            "backbone": self.backbone,
            "target_columns": list(self.target_columns),
            "input_mode": self.input_mode,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            preprocess=PreprocessConfig.from_dict(d["preprocess"]),
            backbone=d["backbone"],
            target_columns=list(d["target_columns"]),
            input_mode=d.get("input_mode", "grey"),
        )
