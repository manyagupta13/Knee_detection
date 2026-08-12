"""The mention probe must be high-recall: it bounds what any labeler can reach."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ceiling import _mention_regex, mention_table
from config import ID_COLUMN, TARGET_COLUMNS
from textlabel import normalize


def test_probe_catches_mentions_across_languages():
    cases = [
        ("Effusion", "Derrame articular moderado."),
        ("Effusion", "Efuzyon izlenmedi."),
        ("Effusion", "Kein Gelenkerguss."),
        ("Fracture", "Kirik saptanmadi."),
        ("PF OA", "Condromalacia rotuliana grado 2."),
        ("Medial Meniscus", "Menisco interno normal."),
        ("Baker's", "Quiste de Baker."),
        ("Synovitis", "Sinovitis leve."),
    ]
    for col, text in cases:
        assert _mention_regex(col).search(normalize(text)), (col, text)


def test_probe_is_silent_when_the_finding_is_absent():
    """A report that never discusses the finding must not count as a mention."""
    text = normalize("The cruciate ligaments are intact.")
    assert not _mention_regex("Baker's").search(text)
    assert not _mention_regex("Fracture").search(text)


def test_unmentionable_gold_positive_is_counted():
    """The whole point: a gold positive the report never discusses is
    irrecoverable, and must be separated from an extraction failure."""
    uids = ["s1", "s2"]
    gold = pd.DataFrame(
        {c: [np.nan, np.nan] for c in TARGET_COLUMNS}, index=uids, dtype=float
    )
    gold.loc["s1", "Fracture"] = 1.0   # report DOES mention it
    gold.loc["s2", "Fracture"] = 1.0   # report does NOT
    reports = pd.DataFrame(
        {
            ID_COLUMN: uids,
            "report_text": [
                "Acute fracture of the tibial plateau.",
                "The cruciate ligaments are intact.",
            ],
        }
    )
    preds = pd.DataFrame({c: [np.nan, np.nan] for c in TARGET_COLUMNS},
                         index=uids, dtype=float)
    preds.loc["s1", "Fracture"] = 1.0

    t = mention_table(gold, reports, preds, ["Fracture"]).set_index("column")
    assert t.loc["Fracture", "n_pos"] == 2
    assert t.loc["Fracture", "mention_rate_pos"] == 0.5
    assert t.loc["Fracture", "missed_unmentionable"] == 1


def test_accuracy_given_mention_ignores_unmentioned_cells():
    uids = ["s1", "s2"]
    gold = pd.DataFrame({c: [np.nan, np.nan] for c in TARGET_COLUMNS},
                        index=uids, dtype=float)
    gold.loc["s1", "Effusion"] = 1.0
    gold.loc["s2", "Effusion"] = 1.0
    reports = pd.DataFrame({
        ID_COLUMN: uids,
        "report_text": ["Joint effusion present.", "Ligaments intact."],
    })
    preds = pd.DataFrame({c: [np.nan, np.nan] for c in TARGET_COLUMNS},
                         index=uids, dtype=float)
    preds.loc["s1", "Effusion"] = 1.0
    preds.loc["s2", "Effusion"] = 0.0  # wrong, but on an unmentioned cell

    t = mention_table(gold, reports, preds, ["Effusion"]).set_index("column")
    assert t.loc["Effusion", "acc_given_mention"] == 1.0
