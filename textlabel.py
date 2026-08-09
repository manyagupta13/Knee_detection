"""Precision-first multilingual pseudo-labeler for TWO columns: Effusion, Fracture.

Text is a supervision channel, never a model input (the test set has no reports).

Phase 0 scope, deliberately narrow (see PHASE0.md):
  * only Effusion and Fracture - every other column stays NaN and is masked out
    of the loss;
  * a study is labeled only on an unambiguous match. Hedged phrasing
    ("trace effusion", "no significant effusion", "possible fracture") abstains
    rather than guessing, because §2.3 of PLAN.md measured that naive keyword +
    negation *systematically* mislabels common phrasing;
  * absence of any mention is NaN, not 0. Absence-of-mention-as-negative has to
    be verified per finding per institution first.

TODO(phase-1): replace this with a fine-tuned multilingual encoder / per-report
LLM extraction covering all 12 columns with soft labels + confidence weights.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd

from config import ID_COLUMN, PSEUDO_LABEL_COLUMNS, TARGET_COLUMNS

LANGUAGES = ("en", "es", "tr", "de", "nl")

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
_CHAR_MAP = {
    "ı": "i",
    "İ": "i",
    "ß": "ss",
    "ø": "o",
    "đ": "d",
    "ł": "l",
}


def normalize(text: str) -> str:
    """Lowercase, strip diacritics, squash punctuation-free comparison issues.

    All patterns below are written against this normalized form, so they must be
    diacritic-free too ("efuzyon", not "efüzyon").
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower()
    s = "".join(_CHAR_MAP.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# Positive term patterns per finding, per language. Hand-translated for the top
# 5 languages by share (en 39%, es 16%, tr 10%, de 6%, nl 3%).
FINDING_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "Effusion": {
        "en": (r"\beffusions?\b", r"\bjoint fluid\b", r"\bintra ?articular fluid\b"),
        "es": (r"\bderrame\w*\b",),
        "tr": (r"\befuzyon\w*\b", r"\beklem sivisi\b"),
        "de": (r"\b\w*erguss\w*\b", r"\bgelenkflussigkeit\b"),
        "nl": (r"\bhydrops\b", r"\bgewrichtsvocht\b", r"\beffusie\b"),
    },
    "Fracture": {
        "en": (r"\bfractur\w*\b",),
        "es": (r"\bfractur\w*\b",),
        "tr": (r"\bkirik\w*\b",),
        "de": (r"\bfraktur\w*\b", r"\bknochenbruch\w*\b"),
        "nl": (r"\bfractu\w*\b", r"\bbotbreuk\w*\b"),
    },
}

# Negation cues. Turkish negates with a suffix *after* the noun
# ("efuzyon izlenmedi"), so cues are matched anywhere inside the clause rather
# than only to the left of the term.
NEGATION_CUES: dict[str, tuple[str, ...]] = {
    "en": (
        r"\bno\b",
        r"\bnot\b",
        r"\bwithout\b",
        r"\babsent\b",
        r"\babsence of\b",
        r"\bfree of\b",
        r"\bnegative for\b",
        r"\bnone\b",
    ),
    "es": (
        r"\bno\b",
        r"\bsin\b",
        r"\bausencia de\b",
        r"\bausente\b",
        r"\bnieg\w*\b",
        r"\bdescarta\w*\b",
    ),
    "tr": (
        r"\byok\w*\b",
        r"\bizlenmemis\w*\b",
        r"\bizlenmedi\b",
        r"\bizlenmemektedir\b",
        r"\bsaptanmam\w*\b",
        r"\bsaptanmadi\b",
        r"\bgorulmedi\b",
        r"\bgorulmemis\w*\b",
        r"\btespit edilmedi\b",
        r"\bmevcut degil\b",
        r"\brastlanmadi\b",
    ),
    "de": (r"\bkein\w*\b", r"\bohne\b", r"\bnicht\b", r"\bfrei von\b", r"\bnegativ\b"),
    "nl": (r"\bgeen\b", r"\bzonder\b", r"\bniet\b", r"\bafwezig\b", r"\bvrij van\b"),
}

# Hedges that force an abstention (NaN) regardless of everything else in the
# clause. This is where the precision comes from.
HEDGE_CUES: tuple[str, ...] = (
    # en
    r"\btrace\b",
    r"\bminimal\w*\b",
    r"\btiny\b",
    r"\bsubtle\b",
    r"\bpossible?\w*\b",
    r"\bposibl\w*\b",
    r"\bprobabl\w*\b",
    r"\bquestionable\b",
    r"\bequivocal\b",
    r"\bborderline\b",
    r"\bsuspicious for\b",
    r"\bcannot (?:be )?exclude\w*\b",
    r"\bcould not be excluded\b",
    r"\bmay represent\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bphysiolog\w*\b",
    r"\bsmall amount\b",
    r"\bcannot be ruled out\b",
    r"\brule out\b",
    r"\bsuspect\w*\b",
    # es
    r"\bminim\w*\b",
    r"\bleve\b",
    r"\bescas\w*\b",
    r"\bdiscret\w*\b",
    r"\bdudos\w*\b",
    r"\bsospech\w*\b",
    r"\bno se puede excluir\b",
    # tr
    r"\bsuphel\w*\b",
    r"\bolasi\b",
    r"\bhafif\b",
    r"\baz miktarda\b",
    r"\bekarte edilemez\b",
    # de
    r"\bgering\w*\b",
    r"\bfraglich\w*\b",
    r"\bmoglich\w*\b",
    r"\bverdacht\w*\b",
    r"\bdiskret\w*\b",
    r"\bnicht auszuschliessen\b",
    r"\bnicht sicher\b",
    # nl
    r"\bgering\w*\b",
    r"\bmogelijk\w*\b",
    r"\btwijfel\w*\b",
    r"\bverdenking\b",
    r"\bniet uit te sluiten\b",
    r"\bweinig\b",
)

# "no *significant* effusion" is not a clean negative (§2.3): a negation cue plus
# a significance modifier in the same clause abstains.
SIGNIFICANCE_CUES: tuple[str, ...] = (
    r"\bsignificant\w*\b",
    r"\bsubstantial\w*\b",
    r"\bappreciable\b",
    r"\bsignificativ\w*\b",
    r"\bimportante\b",
    r"\bbelirgin\w*\b",
    r"\bonemli\b",
    r"\bwesentlich\w*\b",
    r"\brelevant\w*\b",
    r"\bausgepragt\w*\b",
    r"\bduidelijk\w*\b",
    r"\bnoemenswaardig\w*\b",
)

# Clause splitting: sentence enders, list separators, and contrastive
# conjunctions in all five languages ("no fracture, but an effusion").
_CLAUSE_SPLIT = re.compile(
    r"[.;:\n\r]+|,|\bbut\b|\bhowever\b|\bpero\b|\bsin embargo\b|\bancak\b|\bfakat\b"
    r"|\baber\b|\bjedoch\b|\bmaar\b|\bechter\b"
)

# "nondisplaced fracture" / "fractura no desplazada" / "nicht dislozierte Fraktur":
# the negation attaches to a modifier, not to the finding. These spans are
# deleted before negation detection - otherwise every acute fracture in es/de/nl
# reads as a negative, which is exactly the §2.3 failure mode.
NEGATION_EXCEPTIONS: tuple[str, ...] = (
    r"\b(?:no|not|non|nicht|niet|sin|geen|kein\w*)[\s-]*"
    r"(?:displaced|desplazad\w*|disloc\w*|disloz\w*|gedisloceerd\w*|deplase\w*|"
    r"gedislokeerd\w*|dislokas\w*)\b",
    r"\bnon[\s-]?displaced\b",
)

_NEG_EXCEPTION_RE = re.compile("|".join(NEGATION_EXCEPTIONS))
_HEDGE_RE = re.compile("|".join(HEDGE_CUES))
_SIGNIF_RE = re.compile("|".join(SIGNIFICANCE_CUES))
_NEG_RE = re.compile("|".join(p for cues in NEGATION_CUES.values() for p in cues))
_TERM_RE = {
    finding: re.compile("|".join(p for pats in langs.values() for p in pats))
    for finding, langs in FINDING_TERMS.items()
}


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(normalize(text)) if c.strip()]


def label_report(text: str) -> dict[str, float | None]:
    """Label one report. Returns {finding: 1.0 | 0.0 | None} for Phase-0 columns.

    ``None`` means "no opinion" and becomes NaN + mask=False downstream.
    """
    out: dict[str, float | None] = {f: None for f in PSEUDO_LABEL_COLUMNS}
    clauses = split_clauses(text)
    for finding in PSEUDO_LABEL_COLUMNS:
        term_re = _TERM_RE[finding]
        positives = 0
        negatives = 0
        for clause in clauses:
            if not term_re.search(clause):
                continue
            negated = bool(_NEG_RE.search(_NEG_EXCEPTION_RE.sub(" ", clause)))
            if _HEDGE_RE.search(clause):
                continue  # hedged -> this clause casts no vote
            if negated and _SIGNIF_RE.search(clause):
                continue  # "no significant effusion" -> abstain
            if negated:
                negatives += 1
            else:
                positives += 1
        if positives and not negatives:
            out[finding] = 1.0
        elif negatives and not positives:
            out[finding] = 0.0
        else:
            out[finding] = None  # nothing found, or a contradiction
    return out


def label_reports(
    reports: pd.DataFrame,
    text_col: str = "report_text",
    id_col: str = ID_COLUMN,
    columns: Iterable[str] = TARGET_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label a report table.

    Returns ``(labels, mask)``, both indexed by StudyInstanceUID with all 12
    target columns. ``labels`` holds floats with NaN where unknown; ``mask`` is
    the per-(study, column) boolean the masked BCE consumes.
    """
    columns = list(columns)
    index = pd.Index(reports[id_col].astype(str), name=id_col)
    labels = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    for uid, text in zip(index, reports[text_col]):
        for finding, value in label_report(text).items():
            if value is not None and finding in labels.columns:
                labels.loc[uid, finding] = value
    mask = labels.notna()
    return labels, mask


def coverage_report(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-column counts of pseudo-labels produced - log this every run."""
    rows = []
    for col in labels.columns:
        s = labels[col]
        rows.append(
            {
                "column": col,
                "labeled": int(s.notna().sum()),
                "positive": int((s == 1).sum()),
                "negative": int((s == 0).sum()),
            }
        )
    return pd.DataFrame(rows)
