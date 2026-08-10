"""The pseudo-labeler is precision-first: it must abstain rather than guess."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import PSEUDO_LABEL_COLUMNS, TARGET_COLUMNS
from data import load_tables
from textlabel import LANGUAGES, coverage_report, label_report, label_reports, normalize

POSITIVES = {
    "en": "There is a moderate joint effusion in the suprapatellar recess.",
    "es": "Se observa derrame articular moderado.",
    "tr": "Eklem aralIgInda belirgin efuzyon mevcuttur.",
    "de": "Deutlicher Kniegelenkserguss nachweisbar.",
    "nl": "Er is duidelijke hydrops van het kniegewricht.",
}
NEGATIVES = {
    "en": "No joint effusion is identified.",
    "es": "No se identifica derrame articular.",
    "tr": "Efuzyon izlenmedi.",
    "de": "Kein Gelenkerguss nachweisbar.",
    "nl": "Geen gewrichtsvocht aanwezig.",
}
HEDGES = {
    "en": "No significant effusion; trace fluid may be physiologic.",
    "es": "Sin derrame significativo; minima cantidad de liquido.",
    "tr": "Belirgin efuzyon izlenmemektedir, minimal sivi mevcut.",
    "de": "Kein wesentlicher Erguss, geringe Flussigkeitsmenge.",
    "nl": "Geen duidelijke hydrops, geringe hoeveelheid vocht.",
}

FRACTURE_POSITIVES = {
    "en": "Acute nondisplaced fracture of the lateral tibial plateau.",
    "es": "Fractura no desplazada del platillo tibial lateral.",
    "tr": "Lateral tibia platosunda kirik hatti izlenmektedir.",
    "de": "Nicht dislozierte Fraktur des lateralen Tibiaplateaus.",
    "nl": "Niet-gedisloceerde fractuur van het laterale tibiaplateau.",
}
FRACTURE_NEGATIVES = {
    "en": "No fracture is seen.",
    "es": "Sin fractura.",
    "tr": "Kirik saptanmadi.",
    "de": "Keine Fraktur.",
    "nl": "Geen fractuur zichtbaar.",
}
FRACTURE_HEDGES = {
    "en": "Possible occult fracture, cannot be excluded.",
    "es": "Posible fractura oculta, dudoso.",
    "tr": "Suphel kirik hatti, ekarte edilemez.",
    "de": "Fragliche Fraktur, nicht sicher abgrenzbar.",
    "nl": "Mogelijk occulte fractuur, niet uit te sluiten.",
}


def test_normalize_strips_diacritics():
    assert normalize("Efüzyon KIRIK") == "efuzyon kirik"
    assert normalize("Kniegelenksergüsse") == "kniegelenksergusse"
    assert normalize(None) == ""


@pytest.mark.parametrize("lang", LANGUAGES)
def test_effusion_positive(lang):
    assert label_report(POSITIVES[lang])["Effusion"] == 1.0


@pytest.mark.parametrize("lang", LANGUAGES)
def test_effusion_negative(lang):
    assert label_report(NEGATIVES[lang])["Effusion"] == 0.0


@pytest.mark.parametrize("lang", LANGUAGES)
def test_effusion_hedge_abstains(lang):
    assert label_report(HEDGES[lang])["Effusion"] is None


@pytest.mark.parametrize("lang", LANGUAGES)
def test_fracture_positive(lang):
    assert label_report(FRACTURE_POSITIVES[lang])["Fracture"] == 1.0


@pytest.mark.parametrize("lang", LANGUAGES)
def test_fracture_negative(lang):
    assert label_report(FRACTURE_NEGATIVES[lang])["Fracture"] == 0.0


@pytest.mark.parametrize("lang", LANGUAGES)
def test_fracture_hedge_abstains(lang):
    assert label_report(FRACTURE_HEDGES[lang])["Fracture"] is None


def test_absence_of_mention_is_nan_not_zero():
    """Absence-of-mention-as-negative is unverified (PLAN.md §2.3)."""
    out = label_report("The cruciate ligaments and collateral ligaments are intact.")
    assert out["Effusion"] is None
    assert out["Fracture"] is None


def test_contradiction_abstains():
    text = "Joint effusion is present. No joint effusion is identified."
    assert label_report(text)["Effusion"] is None


def test_contrastive_clause_is_split():
    out = label_report("No fracture is seen, but there is a moderate joint effusion.")
    assert out["Fracture"] == 0.0
    assert out["Effusion"] == 1.0


def test_all_twelve_columns_are_produced():
    """Phase 1 widened the labeler from 2 columns to all 12."""
    assert set(label_report(POSITIVES["en"])) == set(TARGET_COLUMNS)


def test_columns_argument_restricts_output():
    out = label_report(POSITIVES["en"], columns=["Effusion"])
    assert set(out) == {"Effusion"}


def test_label_reports_emits_full_width_frame_and_mask(fixture_dir):
    tables = load_tables(fixture_dir, split="train")
    labels, mask = label_reports(tables.reports)

    assert list(labels.columns) == list(TARGET_COLUMNS)
    assert mask.shape == labels.shape
    assert (mask == labels.notna()).to_numpy().all()

    out_of_scope = [c for c in TARGET_COLUMNS if c not in PSEUDO_LABEL_COLUMNS]
    assert labels[out_of_scope].isna().to_numpy().all(), "Phase 0 labels 2 columns only"
    assert mask[list(PSEUDO_LABEL_COLUMNS)].to_numpy().any()


def test_precision_against_fixture_ground_truth(fixture_dir):
    """Every label the labeler *does* emit on the fixture must be correct."""
    tables = load_tables(fixture_dir, split="train")
    labels, _ = label_reports(tables.reports)
    gold = tables.gold.set_index(tables.gold["StudyInstanceUID"].astype(str))

    checked = 0
    for uid in gold.index:
        for col in PSEUDO_LABEL_COLUMNS:
            pred = labels.loc[uid, col]
            truth = float(gold.loc[uid, col])
            # train.csv now has a row per study; most carry no gold label at all,
            # and there is nothing to check those pseudo-labels against.
            if np.isnan(pred) or np.isnan(truth):
                continue
            assert pred == truth, (uid, col)
            checked += 1
    assert checked > 0


def test_coverage_report_shape(fixture_dir):
    tables = load_tables(fixture_dir, split="train")
    labels, _ = label_reports(tables.reports)
    cov = coverage_report(labels)
    assert list(cov.columns) == ["column", "labeled", "positive", "negative"]
    assert len(cov) == len(TARGET_COLUMNS)
    eff = cov.set_index("column").loc["Effusion"]
    assert eff["positive"] > 0 and eff["negative"] > 0


# ---------------------------------------------------------------------------
# Phase 1: structure+cue findings and laterality anchoring
# ---------------------------------------------------------------------------
def test_structure_mention_alone_is_not_a_finding():
    """PLAN.md §2.3: the medial meniscus is named in 62% of reports where it is
    NORMAL. Naming the structure must prove nothing on its own."""
    out = label_report("The medial meniscus is visualized on sagittal images.")
    assert out["Medial Meniscus"] is None


def test_structure_with_tear_cue_is_positive():
    out = label_report("There is a tear of the medial meniscus posterior horn.")
    assert out["Medial Meniscus"] == 1.0


def test_structure_with_integrity_cue_is_negative():
    out = label_report("The medial meniscus is intact.")
    assert out["Medial Meniscus"] == 0.0


def test_laterality_is_anchored_not_bag_of_words():
    """The single most dangerous failure mode: medial and lateral are separate
    scored columns, so a tear on one must never label the other."""
    out = label_report("The medial meniscus is intact. The lateral meniscus is torn.")
    assert out["Lateral Meniscus"] == 1.0
    assert out["Medial Meniscus"] == 0.0


@pytest.mark.parametrize(
    "text,column",
    [
        ("Rotura del menisco interno.", "Medial Meniscus"),
        ("Innenmeniskusriss nachweisbar.", "Medial Meniscus"),
        ("Dis menisk yirtigi izlenmektedir.", "Lateral Meniscus"),
        ("Ruptuur van de voorste kruisband.", "ACL"),
        ("Complete tear of the anterior cruciate ligament.", "ACL"),
        ("Quiste de Baker en el hueco popliteo.", "Baker's"),
        ("Bakerzyste nachweisbar.", "Baker's"),
        ("Sinovitis con engrosamiento sinovial.", "Synovitis"),
        ("Bone marrow edema in the lateral femoral condyle.", "Contusion"),
        ("Chondromalacia patellae grade 3.", "PF OA"),
        ("Medial compartment osteoarthritis with osteophytes.", "Medial OA"),
    ],
)
def test_multilingual_positive_examples(text, column):
    assert label_report(text)[column] == 1.0, (text, column)


def test_acl_intact_is_a_clean_negative():
    for text in [
        "The anterior cruciate ligament is intact.",
        "Ligamento cruzado anterior integro.",
        "Vorderes Kreuzband intakt.",
    ]:
        assert label_report(text)["ACL"] == 0.0, text


def test_degeneration_is_not_a_meniscal_tear():
    """Degenerative signal != tear. Conflating them manufactures false
    positives on the two meniscus columns."""
    out = label_report("Degenerative signal within the medial meniscus without tear.")
    assert out["Medial Meniscus"] != 1.0


def test_empty_and_garbage_input():
    for text in ["", None, float("nan"), "12345 !!! ---"]:
        assert label_report(text) == {c: None for c in TARGET_COLUMNS}
