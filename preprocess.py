"""DICOM -> model tensor. ONE code path, used identically for train and test.

Design invariant #1 (see PLAN.md): ``preprocess_study`` always returns the
stacked *multi-series* form, even when called with ``max_series=1``. Widening to
real multi-series fusion in Phase 2 is then a config change, not a rewrite.
There is deliberately no test-only branch anywhere in this module.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from config import DEFAULT_PLANE_PREFS

# ---------------------------------------------------------------------------
# Series description parsing (multilingual, deliberately shallow for Phase 0)
# ---------------------------------------------------------------------------
_PLANE_PATTERNS: dict[str, tuple[str, ...]] = {
    # en / es / tr / de / nl variants plus the usual scanner abbreviations
    "sagittal": (r"\bsag\w*", r"\bsg\b"),
    "coronal": (r"\bcor\w*", r"\bkor\w*", r"\bcr\b"),
    "axial": (r"\bax\w*", r"\btra\w*", r"\btransvers\w*"),
}

# Fluid-sensitive: T2 / PD / STIR / TIRM / any fat-suppressed acquisition.
_FLUID_PATTERNS: tuple[str, ...] = (
    r"\bt2\b",
    r"\bt2\*",
    r"\bstir\b",
    r"\btirm\b",
    r"\bpd\w*",
    r"\bdp\b",  # es: densidad protonica
    r"\bfs\b",
    r"\bfatsat\b",
    r"\bfat[\s_-]?sat\w*",
    r"\bspair\b",
    r"\bspir\b",
    r"\bsti\b",
)
# T1 without fat-sat is explicitly not fluid-sensitive.
_T1_PATTERN = r"\bt1\b"

# Unit normals of the three canonical planes in patient (LPS) coordinates.
_PLANE_NORMALS = {
    "sagittal": np.array([1.0, 0.0, 0.0]),
    "coronal": np.array([0.0, 1.0, 0.0]),
    "axial": np.array([0.0, 0.0, 1.0]),
}


class StudyVolume(NamedTuple):
    """Stacked multi-series representation of one study.

    pixels:      uint8, shape (max_series, n_slices, size, size)
    series_mask: bool,  shape (max_series,) - True where a real series was found
    series_uids: list[str | None], length max_series
    """

    pixels: np.ndarray
    series_mask: np.ndarray
    series_uids: list[str | None]


def _norm_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9*]+", " ", str(value).lower()).strip()


def classify_plane_from_iop(iop: Sequence[float] | None) -> str:
    """Plane from ImageOrientationPatient (row/col direction cosines)."""
    if iop is None or len(iop) != 6:
        return "unknown"
    row = np.asarray(iop[:3], dtype=float)
    col = np.asarray(iop[3:], dtype=float)
    normal = np.cross(row, col)
    n = np.linalg.norm(normal)
    if n == 0:
        return "unknown"
    normal = normal / n
    scores = {p: abs(float(normal @ v)) for p, v in _PLANE_NORMALS.items()}
    plane = max(scores, key=scores.get)
    return plane if scores[plane] >= 0.5 else "unknown"


def classify_series(description, iop: Sequence[float] | None = None) -> tuple[str, str]:
    """Return ``(plane, contrast)`` where contrast is 'fluid' or 'other'.

    Description wins when it is unambiguous; ImageOrientationPatient is the
    fallback (and the more trustworthy signal, hence Phase 2 will flip the
    priority once orientation canonicalization lands).
    """
    text = _norm_text(description)
    plane = "unknown"
    for candidate, patterns in _PLANE_PATTERNS.items():
        if any(re.search(p, text) for p in patterns):
            plane = candidate
            break
    if plane == "unknown":
        plane = classify_plane_from_iop(iop)

    is_fluid = any(re.search(p, text) for p in _FLUID_PATTERNS)
    if is_fluid and re.search(_T1_PATTERN, text) and not re.search(r"fs|fat|spair|spir", text):
        is_fluid = False
    return plane, ("fluid" if is_fluid else "other")


def _pref_matches(pref: str, plane: str, contrast: str) -> bool:
    want_plane, _, want_contrast = pref.partition("_")
    if want_plane != "any" and want_plane != plane:
        return False
    if want_contrast in ("", "any"):
        return True
    return want_contrast == contrast


# ---------------------------------------------------------------------------
# Series table helpers
# ---------------------------------------------------------------------------
def attach_series_dirs(series_df: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Add a ``series_dir`` column: ``root/<StudyInstanceUID>/<SeriesInstanceUID>``.

    Used identically for train and test; only ``root`` differs.
    """
    root = Path(root)
    out = series_df.copy()
    out["series_dir"] = [
        str(root / str(s) / str(se))
        for s, se in zip(out["StudyInstanceUID"], out["SeriesInstanceUID"])
    ]
    return out


def _list_dicoms(series_dir: str | Path) -> list[Path]:
    d = Path(series_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))


def _peek_iop(series_dir: str | Path) -> list[float] | None:
    files = _list_dicoms(series_dir)
    if not files:
        return None
    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    except Exception:
        return None
    iop = getattr(ds, "ImageOrientationPatient", None)
    return [float(v) for v in iop] if iop is not None else None


def select_series(
    study_series: pd.DataFrame,
    plane_prefs: Sequence[str] = DEFAULT_PLANE_PREFS,
    max_series: int = 1,
) -> list[dict]:
    """Pick up to ``max_series`` series for a study, honouring ``plane_prefs``.

    First-match-wins over the preference list; ties broken by SeriesInstanceUID
    for determinism. Returns a list of row dicts (may be shorter than
    ``max_series`` when the study is missing everything we want).
    """
    if study_series.empty:
        return []

    rows = study_series.sort_values("SeriesInstanceUID", kind="stable").to_dict("records")
    classified = []
    for row in rows:
        plane, contrast = classify_series(row.get("SeriesDescription"))
        if plane == "unknown" and row.get("series_dir"):
            plane = classify_plane_from_iop(_peek_iop(row["series_dir"]))
        classified.append((row, plane, contrast))

    chosen: list[dict] = []
    seen: set[str] = set()
    for pref in list(plane_prefs) + ["any_any"]:  # last resort: take anything
        for row, plane, contrast in classified:
            if len(chosen) >= max_series:
                return chosen
            uid = str(row["SeriesInstanceUID"])
            if uid in seen or not _pref_matches(pref, plane, contrast):
                continue
            seen.add(uid)
            chosen.append(row)
    return chosen[:max_series]


# ---------------------------------------------------------------------------
# Slice ordering + intensity
# ---------------------------------------------------------------------------
def _slice_sort_key(ds, index: int) -> tuple[float, float]:
    """Project ImagePositionPatient onto the slice normal.

    InstanceNumber lies (interleaved acquisitions, re-sorted archives), so it is
    only the fallback.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
        row = np.asarray([float(v) for v in iop[:3]])
        col = np.asarray([float(v) for v in iop[3:]])
        normal = np.cross(row, col)
        n = np.linalg.norm(normal)
        if n > 0:
            pos = np.asarray([float(v) for v in ipp])
            return (0.0, float(pos @ (normal / n)))
    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        return (1.0, float(inst))
    return (2.0, float(index))


def _sample_indices(n_available: int, n_slices: int) -> list[int]:
    """Evenly spaced slice indices, repeating edges when the series is short."""
    if n_available <= 0:
        return []
    if n_available == 1:
        return [0] * n_slices
    return [int(round(i)) for i in np.linspace(0, n_available - 1, n_slices)]


def _to_uint8(volume: np.ndarray, clip_percentiles: tuple[float, float]) -> np.ndarray:
    """Percentile-clip and scale over the WHOLE series, never per slice.

    Per-slice normalization destroys effusion/fluid cues by re-stretching every
    slice to full range.
    """
    lo_p, hi_p = clip_percentiles
    lo = float(np.percentile(volume, lo_p))
    hi = float(np.percentile(volume, hi_p))
    if hi <= lo:
        hi = float(volume.max())
        lo = float(volume.min())
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.uint8)
    out = (np.clip(volume, lo, hi) - lo) / (hi - lo)
    return (out * 255.0).round().astype(np.uint8)


def _read_pixels(path: Path) -> np.ndarray | None:
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    if arr.ndim == 3:  # multi-frame: collapse to the middle frame for Phase 0
        arr = arr[arr.shape[0] // 2]
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr * slope + intercept


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.shape == (size, size):
        return arr
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))


def preprocess_series(
    series_dir: str | Path,
    n_slices: int,
    size: int,
    clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray | None:
    """One series -> uint8 (n_slices, size, size), or None if unreadable."""
    files = _list_dicoms(series_dir)
    if not files:
        return None

    headers = []
    for i, path in enumerate(files):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        headers.append((_slice_sort_key(ds, i), path))
    if not headers:
        return None
    headers.sort(key=lambda t: t[0])
    ordered = [path for _, path in headers]

    picked = [ordered[i] for i in _sample_indices(len(ordered), n_slices)]

    planes: list[np.ndarray] = []
    for path in picked:
        arr = _read_pixels(path)
        if arr is None:
            planes.append(None)  # type: ignore[arg-type]
        else:
            planes.append(arr)
    real = [a for a in planes if a is not None]
    if not real:
        return None
    fill = np.zeros_like(real[0])
    planes = [fill if a is None else a for a in planes]

    # Common in-plane shape may vary across slices; normalize intensity on the
    # raw values first (per series), then resize.
    flat = np.concatenate([a.ravel() for a in planes])
    lo_p, hi_p = clip_percentiles
    lo = float(np.percentile(flat, lo_p))
    hi = float(np.percentile(flat, hi_p))
    scaled = []
    for a in planes:
        if hi > lo:
            v = (np.clip(a, lo, hi) - lo) / (hi - lo)
            v = (v * 255.0).round().astype(np.uint8)
        else:
            v = _to_uint8(a, clip_percentiles)
        scaled.append(_resize(v, size))
    return np.stack(scaled, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------
def preprocess_study(
    study_uid: str,
    series_df: pd.DataFrame,
    plane_prefs: Sequence[str],
    n_slices: int,
    size: int,
    max_series: int,
) -> StudyVolume:
    """Preprocess one study into the stacked multi-series form.

    ``series_df`` is the full series table (train_series.csv / test_series.csv)
    with at least ``StudyInstanceUID``, ``SeriesInstanceUID``,
    ``SeriesDescription`` and ``series_dir`` (see :func:`attach_series_dirs`).

    Always returns ``pixels`` of shape ``(max_series, n_slices, size, size)``,
    zero-filled and masked where a series is missing - including when
    ``max_series == 1``. Do not add a shortcut that returns 3D here.
    """
    study_series = series_df[series_df["StudyInstanceUID"].astype(str) == str(study_uid)]
    chosen = select_series(study_series, plane_prefs=plane_prefs, max_series=max_series)

    pixels = np.zeros((max_series, n_slices, size, size), dtype=np.uint8)
    series_mask = np.zeros((max_series,), dtype=bool)
    series_uids: list[str | None] = [None] * max_series

    slot = 0
    for row in chosen:
        if slot >= max_series:
            break
        vol = preprocess_series(row.get("series_dir", ""), n_slices=n_slices, size=size)
        if vol is None:
            continue
        pixels[slot] = vol
        series_mask[slot] = True
        series_uids[slot] = str(row["SeriesInstanceUID"])
        slot += 1

    return StudyVolume(pixels=pixels, series_mask=series_mask, series_uids=series_uids)


def to_model_input(pixels: np.ndarray) -> np.ndarray:
    """(n_slices, H, W) uint8 -> (n_slices, 3, H, W) float32 in [0, 1].

    Phase 0 replicates the grey channel. TODO(phase-3): stack 3 adjacent slices
    as channels (2.5D) instead of replicating.
    """
    x = pixels.astype(np.float32) / 255.0
    return np.repeat(x[:, None, :, :], 3, axis=1)


def series_coverage(
    study_uids: Iterable[str],
    series_df: pd.DataFrame,
    plane_prefs: Sequence[str],
    max_series: int,
) -> tuple[int, int]:
    """(studies with >=1 resolvable series, total studies) — no DICOM decode.

    Cheap early-warning check: if this comes back near 0/N, every downstream
    prediction will collapse to a near-constant, input-independent output
    (all-zero pixels in, same logits out for every study), which reads as
    "the model isn't learning" when the real bug is series selection or a
    wrong image root.
    """
    uids = list(study_uids)
    hits = 0
    for uid in uids:
        study_series = series_df[series_df["StudyInstanceUID"].astype(str) == str(uid)]
        if select_series(study_series, plane_prefs=plane_prefs, max_series=max_series):
            hits += 1
    return hits, len(uids)


def iter_study_uids(series_df: pd.DataFrame) -> Iterable[str]:
    seen: set[str] = set()
    for uid in series_df["StudyInstanceUID"].astype(str):
        if uid not in seen:
            seen.add(uid)
            yield uid
