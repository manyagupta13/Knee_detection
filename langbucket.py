"""Cheap language bucketing for reports. No dependency, no download.

PLAN.md §2.2: English is only 39% of reports, and per-language validation is an
exit criterion for Phase 1 - a labeler can look fine on the macro number while
being useless in Turkish. This assigns each report a language so agreement can
be sliced by it.

Deliberately not a general-purpose detector: it only needs to separate the ~10
languages in this corpus, and clinical reports are full of distinctive function
words. Script detection settles Greek and Cyrillic outright; the Latin-script
languages are scored on stopword hits.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Function words / high-frequency report tokens per language. Chosen to be
# maximally discriminative between the confusable pairs (es/pt, de/nl, hr/pl).
_STOPWORDS: dict[str, tuple[str, ...]] = {
    "en": (
        "the", "and", "is", "of", "with", "no", "there", "are", "normal", "left",
        "right", "knee", "joint", "signal", "intact", "seen", "within", "mild",
    ),
    "es": (
        "de", "la", "el", "en", "se", "con", "los", "las", "del", "por", "un",
        "una", "rodilla", "observa", "signos", "sin", "articular", "aprecia",
    ),
    "tr": (
        "ve", "ile", "bir", "olan", "izlenmektedir", "mevcut", "diz", "eklem",
        "sinyal", "artis", "yoktur", "seviyesinde", "dogal", "olarak", "bulgu",
    ),
    "de": (
        "der", "die", "das", "und", "mit", "kein", "keine", "nicht", "im", "des",
        "bei", "kniegelenk", "unauffallig", "regelrecht", "nachweisbar", "zeigt",
    ),
    "nl": (
        "de", "het", "een", "en", "met", "geen", "van", "is", "zijn", "knie",
        "gewricht", "normale", "aanwezig", "wordt", "bij", "geringe",
    ),
    "fr": (
        "le", "la", "les", "des", "du", "avec", "pas", "une", "est", "genou",
        "articulaire", "sans", "signal", "aspect", "normale",
    ),
    "pl": (
        "w", "i", "na", "nie", "z", "do", "jest", "kolana", "stawu", "obraz",
        "bez", "oraz", "widoczne", "prawego",
    ),
    "hr": (
        "je", "se", "na", "u", "koljena", "bez", "sa", "od", "prikaz", "uredan",
        "zgloba", "nema", "lijevog", "desnog",
    ),
    "it": (
        "di", "il", "la", "del", "con", "non", "ginocchio", "articolare", "che",
        "una", "nella", "presenza",
    ),
    "pt": (
        "de", "da", "do", "com", "nao", "joelho", "articular", "sem", "que",
        "uma", "sinais",
    ),
}

# Tokens that, when present, strongly override the stopword vote. "geen"/"knie"
# are unambiguous Dutch; "kein"/"kniegelenk" unambiguous German - and those two
# languages otherwise share enough short words to get confused.
_STRONG_MARKERS: dict[str, tuple[str, ...]] = {
    "nl": ("geen", "knie", "kraakbeen", "gewrichtsvocht", "kruisband", "meniscus"),
    "de": ("kein", "keine", "kniegelenk", "knorpel", "kreuzband", "erguss"),
    "tr": ("izlenmektedir", "izlenmedi", "saptanmadi", "menisküs", "menisk", "eklem"),
    "es": ("rodilla", "menisco", "derrame", "ligamento", "observa"),
    "en": ("meniscus", "effusion", "ligament", "tear", "unremarkable"),
}

_WORD_RE = re.compile(r"[a-z]+")


def _fold(text: str) -> str:
    s = str(text).lower()
    s = s.replace("ı", "i").replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _script(text: str) -> str | None:
    """Greek and Cyrillic are decided by codepoint, not vocabulary."""
    counts = Counter()
    for ch in str(text):
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if name.startswith("GREEK"):
            counts["el"] += 1
        elif name.startswith("CYRILLIC"):
            counts["ru"] += 1
        else:
            counts["latin"] += 1
    if not counts:
        return None
    top, n = counts.most_common(1)[0]
    return None if top == "latin" else top


def detect_language(text: str, default: str = "unknown") -> str:
    """Best-effort language code for one report.

    Returns an ISO-639-1-ish code, or ``default`` when the text is too short or
    too ambiguous to call. The Slavic/Balkan cluster is approximate (PLAN.md
    §2.2 flags this); treat 'hr' as "some South Slavic language".
    """
    if text is None:
        return default
    script = _script(text)
    if script is not None:
        return script

    folded = _fold(text)
    tokens = _WORD_RE.findall(folded)
    if len(tokens) < 3:
        return default
    seen = set(tokens)

    scores = {
        lang: sum(1 for w in words if w in seen)
        for lang, words in _STOPWORDS.items()
    }
    # Strong markers are worth several ordinary stopword hits.
    for lang, markers in _STRONG_MARKERS.items():
        scores[lang] = scores.get(lang, 0) + 3 * sum(1 for m in markers if _fold(m) in seen)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return default
    # Require a margin over the runner-up, else admit we don't know.
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) > 1 and ranked[0] == ranked[1]:
        return default
    return best


def bucket_languages(texts) -> list[str]:
    return [detect_language(t) for t in texts]
