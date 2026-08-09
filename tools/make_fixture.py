"""Emit a synthetic knee-MRI fixture in the real schema.

~12 fake studies of genuine DICOMs (pydicom) plus train.csv / train_reports.csv /
train_series.csv, a held-out "test" split with sample_submission.csv, mixed
planes and five report languages. Everything in this repo is expected to run
end-to-end on this fixture under pytest, with no Kaggle data present.

Deliberate traps baked in, because these are the bugs that actually happen:
  * InstanceNumber is shuffled relative to true geometric order, so anything
    sorting by InstanceNumber produces scrambled volumes;
  * in-plane matrix size varies between series;
  * some studies have no sagittal fluid-sensitive series, exercising the
    sagittal-any fallback;
  * reports include clean positives, clean negations, and hedged phrasing that
    the labeler must abstain on.

Usage::

    python tools/make_fixture.py --out fixtures/synthetic --n-studies 12
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ID_COLUMN, TARGET_COLUMNS  # noqa: E402

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"

PLANE_IOP = {
    "sagittal": [0, 1, 0, 0, 0, -1],
    "coronal": [1, 0, 0, 0, 0, -1],
    "axial": [1, 0, 0, 0, 1, 0],
}

# (SeriesDescription, plane, is_fluid_sensitive)
SERIES_MENU: list[tuple[str, str, bool]] = [
    ("SAG PD FS", "sagittal", True),
    ("SAG T2 TSE", "sagittal", True),
    ("SAG STIR", "sagittal", True),
    ("SAG T1 TSE", "sagittal", False),
    ("COR STIR", "coronal", True),
    ("COR T1", "coronal", False),
    ("AX T2 FS", "axial", True),
    ("Sagital DP SPAIR", "sagittal", True),  # es
    ("KORONAL TIRM", "coronal", True),  # tr/de
    ("TRA T2 FS", "axial", True),  # nl/de
]

# Report templates per language: (positive, negative, hedged-so-abstain)
REPORTS = {
    "en": {
        "effusion_pos": "There is a moderate joint effusion in the suprapatellar recess.",
        "effusion_neg": "No joint effusion is identified.",
        "effusion_hedge": "No significant effusion; trace fluid may be physiologic.",
        "fracture_pos": "Acute nondisplaced fracture of the lateral tibial plateau.",
        "fracture_neg": "No fracture is seen.",
        "fracture_hedge": "Possible occult fracture, cannot be excluded.",
        "filler": "The cruciate ligaments and collateral ligaments are intact.",
    },
    "es": {
        "effusion_pos": "Se observa derrame articular moderado en el receso suprarrotuliano.",
        "effusion_neg": "No se identifica derrame articular.",
        "effusion_hedge": "Sin derrame significativo; minima cantidad de liquido.",
        "fracture_pos": "Fractura no desplazada del platillo tibial lateral.",
        "fracture_neg": "Sin fractura.",
        "fracture_hedge": "Posible fractura oculta, dudoso.",
        "filler": "Ligamentos cruzados y colaterales integros.",
    },
    "tr": {
        "effusion_pos": "Eklem aralIgInda belirgin efuzyon mevcuttur.",
        "effusion_neg": "Efuzyon izlenmedi.",
        "effusion_hedge": "Belirgin efuzyon izlenmemektedir, minimal sivi mevcut.",
        "fracture_pos": "Lateral tibia platosunda kirik hatti izlenmektedir.",
        "fracture_neg": "Kirik saptanmadi.",
        "fracture_hedge": "Suphel kirik hatti, ekarte edilemez.",
        "filler": "Capraz baglar ve kollateral bagler normal gorunumdedir.",
    },
    "de": {
        "effusion_pos": "Deutlicher Kniegelenkserguss im Recessus suprapatellaris.",
        "effusion_neg": "Kein Gelenkerguss nachweisbar.",
        "effusion_hedge": "Kein wesentlicher Erguss, geringe Flussigkeitsmenge.",
        "fracture_pos": "Nicht dislozierte Fraktur des lateralen Tibiaplateaus.",
        "fracture_neg": "Keine Fraktur.",
        "fracture_hedge": "Fragliche Fraktur, nicht sicher abgrenzbar.",
        "filler": "Kreuzbander und Kollateralbander regelrecht.",
    },
    "nl": {
        "effusion_pos": "Er is duidelijke hydrops van het kniegewricht.",
        "effusion_neg": "Geen gewrichtsvocht aanwezig.",
        "effusion_hedge": "Geen duidelijke hydrops, geringe hoeveelheid vocht.",
        "fracture_pos": "Niet-gedisloceerde fractuur van het laterale tibiaplateau.",
        "fracture_neg": "Geen fractuur zichtbaar.",
        "fracture_hedge": "Mogelijk occulte fractuur, niet uit te sluiten.",
        "filler": "Kruisbanden en collaterale banden intact.",
    },
}

# Per study: (effusion state, fracture state) where state is pos/neg/hedge/absent
STUDY_PLAN = [
    ("en", "pos", "neg"),
    ("en", "neg", "neg"),
    ("es", "pos", "pos"),
    ("es", "neg", "hedge"),
    ("tr", "pos", "neg"),
    ("tr", "hedge", "neg"),
    ("de", "neg", "pos"),
    ("de", "pos", "absent"),
    ("nl", "neg", "neg"),
    ("nl", "pos", "hedge"),
    ("en", "hedge", "absent"),
    ("es", "pos", "neg"),
]


def _make_slice(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    z: float,
    effusion: bool,
    fracture: bool,
) -> np.ndarray:
    """A crude knee-ish phantom so training has something learnable."""
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    cy, cx = rows / 2, cols / 2
    img = 300 + 120 * np.exp(-(((yy - cy) ** 2) / (2 * (rows / 3) ** 2) + ((xx - cx) ** 2) / (2 * (cols / 3) ** 2)))
    img += rng.normal(0, 12, size=img.shape).astype(np.float32)
    if effusion and abs(z) < 6:  # bright fluid, central slices only
        blob = 900 * np.exp(-(((yy - cy * 0.6) ** 2) + ((xx - cx * 1.2) ** 2)) / 12.0)
        img += blob
    if fracture and abs(z) < 3:  # dark line through the tibial plateau
        img -= 400 * np.exp(-((yy - rows * 0.75) ** 2) / 1.5)
    return np.clip(img, 0, 4095).astype(np.uint16)


def _write_dicom(
    path: Path,
    pixels: np.ndarray,
    study_uid: str,
    series_uid: str,
    series_desc: str,
    plane: str,
    instance_number: int,
    z: float,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MR_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MR_IMAGE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SeriesDescription = series_desc
    ds.Modality = "MR"
    ds.PatientID = study_uid[-8:]
    ds.InstanceNumber = instance_number
    ds.ImageOrientationPatient = [float(v) for v in PLANE_IOP[plane]]
    normal = np.cross(PLANE_IOP[plane][:3], PLANE_IOP[plane][3:]).astype(float)
    ds.ImagePositionPatient = [float(v) for v in (normal * z)]
    ds.SliceThickness = 3.0
    ds.PixelSpacing = [0.4, 0.4]
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), enforce_file_format=True)


def _series_for_study(rng: random.Random, index: int) -> list[tuple[str, str, bool]]:
    """Pick 2-4 series; every 4th study deliberately lacks sagittal-fluid."""
    fluid_sag = [s for s in SERIES_MENU if s[1] == "sagittal" and s[2]]
    other_sag = [s for s in SERIES_MENU if s[1] == "sagittal" and not s[2]]
    non_sag = [s for s in SERIES_MENU if s[1] != "sagittal"]
    if index % 4 == 3:
        chosen = [rng.choice(other_sag)] + rng.sample(non_sag, 2)
    else:
        chosen = [rng.choice(fluid_sag)] + rng.sample(non_sag, rng.choice([1, 2]))
        if rng.random() < 0.5:
            chosen.append(rng.choice(other_sag))
    return chosen


def make_fixture(out_dir: Path, n_studies: int = 12, seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    plan = [STUDY_PLAN[i % len(STUDY_PLAN)] for i in range(n_studies)]

    series_rows: list[dict] = []
    report_rows: list[dict] = []
    gold_rows: list[dict] = []
    study_uids: list[str] = []

    for i, (lang, eff_state, frac_state) in enumerate(plan):
        # UID components may not carry leading zeros.
        study_uid = f"1.2.826.0.1.3680043.9.{seed + 1}.{i + 1}"
        study_uids.append(study_uid)
        effusion_present = eff_state == "pos"
        fracture_present = frac_state == "pos"

        for series_desc, plane, is_fluid in _series_for_study(rng, i):
            series_uid = generate_uid()
            n_sl = rng.choice([6, 8, 11, 14])
            rows_, cols_ = rng.choice([(32, 40), (36, 36), (40, 32)])
            zs = [(k - (n_sl - 1) / 2) * 3.0 for k in range(n_sl)]
            # InstanceNumber deliberately does NOT follow geometry
            instance_numbers = list(range(1, n_sl + 1))
            rng.shuffle(instance_numbers)
            for k, z in enumerate(zs):
                px = _make_slice(
                    nprng,
                    rows_,
                    cols_,
                    z,
                    effusion=effusion_present and is_fluid,
                    fracture=fracture_present,
                )
                _write_dicom(
                    out_dir / "train_series" / study_uid / series_uid / f"{instance_numbers[k]:04d}.dcm",
                    px,
                    study_uid,
                    series_uid,
                    series_desc,
                    plane,
                    instance_numbers[k],
                    z,
                )
            # Real schema: structured plane/contrast flags, no SeriesDescription.
            series_rows.append(
                {
                    ID_COLUMN: study_uid,
                    "SeriesInstanceUID": series_uid,
                    "Fluid_Sensitive": int(is_fluid),
                    "Fat_Suppression": int(is_fluid and "FS" in series_desc.upper()),
                    "Anatomical_Plane": plane.capitalize(),
                }
            )

        tpl = REPORTS[lang]
        parts = [tpl["filler"]]
        if eff_state != "absent":
            parts.append(tpl[f"effusion_{eff_state}"])
        if frac_state != "absent":
            parts.append(tpl[f"fracture_{frac_state}"])
        rng.shuffle(parts)
        report_rows.append(
            {ID_COLUMN: study_uid, "language": lang, "report_text": " ".join(parts)}
        )

        # Gold labels for a minority of studies, mirroring the 1.3% reality.
        if i % 3 == 0:
            row = {ID_COLUMN: study_uid}
            for col in TARGET_COLUMNS:
                if col == "Effusion":
                    row[col] = int(effusion_present)
                elif col == "Fracture":
                    row[col] = int(fracture_present)
                else:
                    row[col] = int(rng.random() < 0.3)
            gold_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series_rows).to_csv(out_dir / "train_series.csv", index=False)
    # train.csv: ONE ROW PER STUDY, carrying the report inline, with the 12 label
    # columns left NaN for every study that isn't gold. This is the real shape -
    # a labels-only table would hide the "row present but unlabeled" case that
    # broke the gold/val split.
    gold_by_uid = {row[ID_COLUMN]: row for row in gold_rows}
    train_rows = []
    for rep in report_rows:
        uid = rep[ID_COLUMN]
        row = {ID_COLUMN: uid, "Report": rep["report_text"]}
        gold = gold_by_uid.get(uid)
        for col in TARGET_COLUMNS:
            row[col] = gold[col] if gold is not None else np.nan
        train_rows.append(row)
    pd.DataFrame(train_rows, columns=[ID_COLUMN, "Report", *TARGET_COLUMNS]).to_csv(
        out_dir / "train.csv", index=False
    )

    # ---- test split: images + series table only. No reports, by design. ----
    test_uids = study_uids[: max(3, n_studies // 3)]
    test_series = pd.DataFrame(series_rows)
    test_series = test_series[test_series[ID_COLUMN].isin(test_uids)].copy()
    for _, r in test_series.iterrows():
        src = out_dir / "train_series" / r[ID_COLUMN] / r["SeriesInstanceUID"]
        dst = out_dir / "test_series" / r[ID_COLUMN] / r["SeriesInstanceUID"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            dst.joinpath(f.name).write_bytes(f.read_bytes())
    test_series.to_csv(out_dir / "test_series.csv", index=False)
    pd.DataFrame({ID_COLUMN: test_uids}).to_csv(out_dir / "test.csv", index=False)

    sample = pd.DataFrame({ID_COLUMN: test_uids})
    for col in TARGET_COLUMNS:
        sample[col] = 0.5
    sample.to_csv(out_dir / "sample_submission.csv", index=False)

    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fixtures/synthetic")
    ap.add_argument("--n-studies", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = make_fixture(Path(args.out), n_studies=args.n_studies, seed=args.seed)
    n_files = sum(1 for _ in out.rglob("*.dcm"))
    print(f"wrote fixture to {out} ({n_files} DICOM files)")
    print(f"pydicom {pydicom.__version__}")


if __name__ == "__main__":
    main()
