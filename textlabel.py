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
from lexicon import (
    CARTILAGE_DAMAGE_WORDS,
    CARTILAGE_WORDS,
    COMPARTMENT_PATTERNS,
    FINDING_SPECS,
    OA_COLUMN_BY_COMPARTMENT,
    OA_EXPLICIT_CUES,
    OA_INTEGRITY_CUES,
    SPEC_BY_COLUMN,
)

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

# Compiled forms of the Phase-1 lexicon.
_SELF_RE = {
    s.column: re.compile("|".join(s.self_sufficient)) if s.self_sufficient else None
    for s in FINDING_SPECS
}
_STRUCT_RE = {
    s.column: re.compile("|".join(s.structures)) if s.structures else None
    for s in FINDING_SPECS
}
_PATH_RE = {
    s.column: re.compile("|".join(s.pathology)) if s.pathology else None
    for s in FINDING_SPECS
}
_INTEG_RE = {
    s.column: re.compile("|".join(s.integrity)) if s.integrity else None
    for s in FINDING_SPECS
}


# Whether "structure discussed, no pathology stated" counts as a negative.
#
# MEASURED per column against gold, and the answer is ACL only:
#
#   column             agreement        recall        coverage
#   ACL                0.929 -> 0.943   1.00 -> 0.94  1137 -> 1941   KEEP
#   MCL                0.875 -> 0.828   1.00 -> 0.40   834 -> 1913   reject
#   Medial Meniscus    0.840 -> 0.767   1.00 -> 0.71  1240 -> 1827   reject
#   Lateral Meniscus   0.900 -> 0.719   0.89 -> 0.47  1460 -> 2137   reject
#
# The collapsing recall is the tell: on the meniscus and collateral columns our
# tear-cue vocabulary misses real tears, so the rule converts a harmless
# abstention into a confident FALSE NEGATIVE - more than half of gold-positive
# lateral meniscus tears end up labeled 0. ACL survives because its tear
# vocabulary is well covered, and there precision actually rose to 0.938 while
# coverage grew 71%.
_MENTION_IMPLIES_NEGATIVE = True
_MENTION_NEGATIVE_COLUMNS: frozenset[str] = frozenset({"ACL"})


def set_mention_implies_negative(
    enabled: bool, columns: Iterable[str] | None = None
) -> None:
    """Treat an un-flagged mention of a structure as evidence it is normal.

    Only sensible for structures a radiologist enumerates whether or not they
    are abnormal - the cruciates, collaterals and menisci. NOT for findings that
    are only ever written down when present (Baker's, Fracture, Contusion),
    where silence means nothing.
    """
    global _MENTION_IMPLIES_NEGATIVE, _MENTION_NEGATIVE_COLUMNS
    _MENTION_IMPLIES_NEGATIVE = bool(enabled)
    if columns is not None:
        _MENTION_NEGATIVE_COLUMNS = frozenset(columns)


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(normalize(text)) if c.strip()]


# ---------------------------------------------------------------------------
# Section-aware parsing (needed for the OA columns)
# ---------------------------------------------------------------------------
_OA_EXPLICIT_RE = re.compile("|".join(OA_EXPLICIT_CUES))
_CARTILAGE_WORD_RE = re.compile("|".join(CARTILAGE_WORDS))
_CARTILAGE_DAMAGE_RE = re.compile("|".join(CARTILAGE_DAMAGE_WORDS))
_OA_INTEGRITY_RE = re.compile("|".join(OA_INTEGRITY_CUES))
_COMPARTMENT_RES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile("|".join(pats))) for name, pats in COMPARTMENT_PATTERNS
)
# Uncertainty abstains everywhere. Severity ("mild", "leve", "gering") does NOT
# abstain for OA - gold counts "mild cartilage thinning" as positive - but does
# for fluid findings, where "trace effusion" is genuinely ambiguous (§2.3).
_UNCERTAINTY_RE = re.compile(
    "|".join(
        p
        for p in HEDGE_CUES
        if not re.search(
            r"trace|minimal|minim|tiny|small amount|leve|escas|discret|gering|hafif|weinig|az miktarda",
            p,
        )
    )
)


def _normalize_lines(text: str) -> list[str]:
    """normalize() collapses newlines, which destroys the section structure the
    OA parser depends on. This keeps line breaks."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = str(text).lower()
    s = "".join(_CHAR_MAP.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.splitlines()]


def compartment_of(text: str) -> str | None:
    """Which knee compartment does this text refer to, if any.

    Patellofemoral is tested first on purpose: "medial patellar facet" belongs
    to the patellofemoral joint, and reading it as the medial compartment would
    write the finding into the wrong scored column.
    """
    for name, rx in _COMPARTMENT_RES:
        if rx.search(text):
            return name
    return None


def _is_header(line: str) -> bool:
    """Section headers look like 'MEDIAL COMPARTMENT:' or 'Medial meniscus:'."""
    return line.endswith(":") and len(line) <= 80


def iter_sections(text: str):
    """Yield ``(compartment, clause)``, carrying compartment across section
    headers so a finding several lines below its header is still attributed."""
    current: str | None = None
    for line in _normalize_lines(text):
        if not line:
            continue
        if _is_header(line):
            found = compartment_of(line)
            if found:
                current = found
            continue  # the header itself asserts no finding
        # An inline "label: value" line ("Medial compartment cartilage: intact")
        # must stay one clause - splitting on the colon separates the structure
        # from its verdict and both halves become uninterpretable.
        inline = line.replace(":", " ")
        if (found := compartment_of(line.split(":", 1)[0])) and ":" in line:
            current = found
        for clause in (c.strip() for c in _CLAUSE_SPLIT.split(inline) if c.strip()):
            yield current, clause


def _label_oa(text: str) -> dict[str, float | None]:
    """Label the three OA columns via compartment attribution."""
    votes: dict[str, dict[str, int]] = {
        col: {"pos": 0, "neg": 0} for col in OA_COLUMN_BY_COMPARTMENT.values()
    }
    for section, clause in iter_sections(text):
        explicit = bool(_OA_EXPLICIT_RE.search(clause))
        about_cartilage = bool(_CARTILAGE_WORD_RE.search(clause))
        damaged = about_cartilage and bool(_CARTILAGE_DAMAGE_RE.search(clause))

        # A bare degeneration word is NOT enough: "medial meniscus ...
        # intrasubstance degeneration" is a meniscal finding, not OA, and it
        # sits under a "Medial compartment:" header.
        if not (explicit or damaged or about_cartilage):
            continue
        if _UNCERTAINTY_RE.search(clause):
            continue  # "possible chondral defect" - no vote

        compartment = compartment_of(clause) or section
        if compartment is None:
            continue
        column = OA_COLUMN_BY_COMPARTMENT[compartment]
        negated = bool(_NEG_RE.search(_NEG_EXCEPTION_RE.sub(" ", clause)))
        if negated and _SIGNIF_RE.search(clause):
            continue

        if explicit or damaged:
            votes[column]["neg" if negated else "pos"] += 1
        elif about_cartilage and _OA_INTEGRITY_RE.search(clause):
            # "Medial compartment cartilage: intact" / "Cartilago rotuliano sin
            # alteraciones" - the negatives the first pass discarded, which is
            # why 95% of emitted OA labels were positive.
            #
            # Deliberately NOT flipped on negation: the common integrity phrases
            # ("sin alteraciones", "geen afwijkingen") carry their own negation,
            # and flipping turned "patellar cartilage unremarkable" into a
            # positive. Rare "not intact" phrasing is sacrificed for that.
            votes[column]["neg"] += 1

    out: dict[str, float | None] = {}
    for column, v in votes.items():
        if v["pos"] and not v["neg"]:
            out[column] = 1.0
        elif v["neg"] and not v["pos"]:
            out[column] = 0.0
        else:
            out[column] = None
    return out


def _near(clause: str, span: tuple[int, int], pattern, window: int) -> bool:
    """Is there a match of ``pattern`` within ``window`` chars of ``span``?

    This is what anchors a tear cue to the structure it belongs to, so that
    "medial meniscus intact, lateral meniscus torn" does not label both.
    """
    if pattern is None:
        return False
    lo = max(0, span[0] - window)
    hi = min(len(clause), span[1] + window)
    return bool(pattern.search(clause, lo, hi))


def _vote_for_clause(column: str, clause: str, spec) -> str | None:
    """One clause's vote for one finding: 'pos', 'neg', or None (abstain)."""
    negated = bool(_NEG_RE.search(_NEG_EXCEPTION_RE.sub(" ", clause)))
    hedged = bool(_HEDGE_RE.search(clause))

    # --- self-sufficient terms: the term IS the finding ---
    self_re = _SELF_RE.get(column)
    if self_re is not None and self_re.search(clause):
        if hedged:
            return None
        if negated and _SIGNIF_RE.search(clause):
            return None  # "no significant effusion"
        return "neg" if negated else "pos"

    # --- structure + cue: presence of the structure proves nothing ---
    struct_re = _STRUCT_RE.get(column)
    if struct_re is None:
        return None
    match = struct_re.search(clause)
    if match is None:
        return None

    if _near(clause, match.span(), _PATH_RE.get(column), spec.window):
        if hedged:
            return None
        if negated and _SIGNIF_RE.search(clause):
            return None
        return "neg" if negated else "pos"

    if _near(clause, match.span(), _INTEG_RE.get(column), spec.window):
        if hedged:
            return None
        # "not intact" flips an integrity statement back to positive
        return "pos" if negated else "neg"

    # Structure named with neither cue nearby.
    #
    # MEASURED: this abstention is where most of our recall goes. The medial
    # meniscus is mentioned in 96.6% of gold studies but we commit a label on
    # only 43.1%, while being 84% accurate when we do commit. A report that
    # discusses the meniscus at length and never says "tear" - "intrasubstance
    # degeneration not extending to the articular surface", "grade 2 signal" -
    # is describing an INTACT structure, and that is a negative we discard.
    if _MENTION_IMPLIES_NEGATIVE and column in _MENTION_NEGATIVE_COLUMNS:
        return None if hedged else "neg"
    return None


def label_report(text: str, columns: Iterable[str] | None = None) -> dict[str, float | None]:
    """Label one report. Returns {finding: 1.0 | 0.0 | None}.

    ``None`` means "no opinion" and becomes NaN + mask=False downstream, which
    is the whole point: abstaining costs recall, guessing costs precision, and
    only precision is recoverable later.
    """
    wanted = [c for c in (columns if columns is not None else SPEC_BY_COLUMN)]
    out: dict[str, float | None] = {c: None for c in wanted}
    clauses = split_clauses(text)

    # OA columns use section-aware compartment attribution, not clause matching.
    oa_columns = set(OA_COLUMN_BY_COMPARTMENT.values())
    if oa_columns & set(wanted):
        for column, value in _label_oa(text).items():
            if column in out:
                out[column] = value

    for column in wanted:
        if column in oa_columns:
            continue
        spec = SPEC_BY_COLUMN.get(column)
        if spec is None:
            continue
        positives = 0
        negatives = 0
        for clause in clauses:
            vote = _vote_for_clause(column, clause, spec)
            if vote == "pos":
                positives += 1
            elif vote == "neg":
                negatives += 1
        if positives and not negatives:
            out[column] = 1.0
        elif negatives and not positives:
            out[column] = 0.0
        else:
            out[column] = None  # nothing found, or a contradiction
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
    # Build row-wise then assign once; .loc per cell is O(n) reallocation and
    # turns 4,400 reports into minutes.
    records = []
    for text in reports[text_col]:
        row = label_report(text, columns=columns)
        records.append([row.get(c) for c in columns])
    values = pd.DataFrame(records, index=index, columns=columns, dtype=float)
    labels.loc[:, columns] = values
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
