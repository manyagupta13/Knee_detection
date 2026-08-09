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

# Columns the Phase-0 text pseudo-labeler is allowed to touch. Precision first:
# every other column stays NaN and is masked out of the loss.
# TODO(phase-1): extend to the remaining 10 columns via a multilingual encoder.
PSEUDO_LABEL_COLUMNS: tuple[str, ...] = ("Effusion", "Fracture")


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


@dataclass(frozen=True)
class PreprocessConfig:
    """Exact preprocessing settings. Saved next to the weights, reloaded at
    inference time so train and test cannot diverge."""

    n_slices: int = 16
    size: int = 224
    max_series: int = 1  # TODO(phase-2): raise for multi-series fusion
    plane_prefs: tuple[str, ...] = DEFAULT_PLANE_PREFS
    clip_percentiles: tuple[float, float] = (1.0, 99.0)

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
        )


@dataclass
class RunConfig:
    """Everything the inference notebook needs to rebuild the model."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    backbone: str = "efficientnet_b0"
    target_columns: Sequence[str] = TARGET_COLUMNS

    def to_dict(self) -> dict:
        return {
            "preprocess": self.preprocess.to_dict(),
            "backbone": self.backbone,
            "target_columns": list(self.target_columns),
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
        )
