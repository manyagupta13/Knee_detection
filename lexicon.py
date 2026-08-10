"""Multilingual finding lexicon for all 12 target columns.

Two kinds of finding, which need different logic:

*Self-sufficient* — the term IS the finding. "Effusion" mentioned unnegated
means effusion. Covers Effusion, Fracture, Synovitis, Baker's, Contusion.

*Structure + cue* — the term is an anatomical structure that is always
mentioned, present or not. "Medial meniscus" appears in most knee reports; what
decides the label is whether a tear cue or an integrity cue sits next to it.
PLAN.md §2.3 measured exactly this: the medial meniscus is named in 62% of
reports where it is normal. Keyword presence alone would be catastrophic here,
so these findings require a pathology cue within a character window of the
structure, and an integrity cue produces a clean negative.

Laterality is handled by making the compartment part of the structure pattern
("medial meniscus", "menisco interno", "Innenmeniskus") rather than searching
for "medial" and "meniscus" independently — anchoring the modifier to its noun
is where naive approaches swap medial and lateral, which are separate columns.

All patterns are written against textlabel.normalize() output: lowercase, no
diacritics. So "efuzyon", not "efüzyon"; "aussenmeniskus", not "Außenmeniskus".

TODO(phase-1b): the remaining languages (el, ru/bg, hr, fr, pl ≈ 23% of the
corpus) are not covered here. Reports in those languages will simply abstain,
which is correct-but-lossy; evaluate_labeler.py reports coverage per language
so the cost is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FindingSpec:
    column: str
    self_sufficient: tuple[str, ...] = ()
    structures: tuple[str, ...] = ()
    pathology: tuple[str, ...] = ()
    integrity: tuple[str, ...] = ()
    window: int = 80


# ---------------------------------------------------------------------------
# Shared cue sets
# ---------------------------------------------------------------------------
# Tear / rupture, across en/es/tr/de/nl. Note "degeneration" is deliberately NOT
# here: degenerative signal is not a tear, and conflating them is a known way to
# manufacture false positives on the meniscus columns.
TEAR_CUES: tuple[str, ...] = (
    # en
    r"\btears?\b", r"\btorn\b", r"\bruptur\w*\b", r"\bdisrupt\w*\b",
    r"\bdiscontinuit\w*\b", r"\bfull[\s-]?thickness\b", r"\bpartial[\s-]?thickness\b",
    r"\bavuls\w*\b",
    # es
    r"\brotura\w*\b", r"\bdesgarr\w*\b", r"\bdiscontinuidad\b", r"\blesion\b",
    # tr — consonant mutation: yirtik -> yirtigi ("yırtığı"), so both stems.
    r"\byirtik\w*\b", r"\byirtig\w*\b", r"\brupturu?\b", r"\bkopma\w*\b",
    r"\bdevamsizlik\b",
    # de
    r"\brisse?\b", r"\brissig\w*\b", r"\bdurchtrenn\w*\b", r"\bkontinuitatsunterbrech\w*\b",
    # nl
    r"\bscheur\w*\b", r"\bruptuur\w*\b", r"\bgescheurd\w*\b",
)

# Explicitly normal / intact — yields a clean negative for structure findings.
INTEGRITY_CUES: tuple[str, ...] = (
    # en
    r"\bintact\b", r"\bnormal\b", r"\bunremarkable\b", r"\bcontinuous\b",
    r"\bpreserved\b", r"\bno tear\b",
    # es
    r"\bintegr\w*\b", r"\bintact\w*\b", r"\bnormal\w*\b", r"\bconservad\w*\b",
    # tr
    r"\bdogal\b", r"\bnormal\b", r"\bsaglam\b", r"\bintakt\b",
    # de
    r"\bintakt\b", r"\bunauffallig\w*\b", r"\bregelrecht\w*\b", r"\bnormal\w*\b",
    r"\bkontinuierlich\w*\b",
    # nl
    r"\bintact\b", r"\bnormaal\b", r"\bnormale\b", r"\bongestoord\w*\b",
)

# Osteoarthritis / cartilage loss.
DEGENERATION_CUES: tuple[str, ...] = (
    # en
    r"\bosteoarthrit\w*\b", r"\barthros\w*\b", r"\barthrit\w*\b", r"\bosteophyt\w*\b",
    r"\bchondral loss\b", r"\bcartilage loss\b", r"\bcartilage thinning\b",
    r"\bjoint space narrowing\b", r"\bdegenerativ\w*\b", r"\bchondromalac\w*\b",
    # es
    r"\bartros\w*\b", r"\bosteofit\w*\b", r"\bcondropat\w*\b", r"\bcondromalac\w*\b",
    r"\bpinzamiento\b", r"\bdegenerativ\w*\b",
    # tr
    r"\bosteoartrit\w*\b", r"\bartroz\w*\b", r"\bosteofit\w*\b", r"\bkondromalaz\w*\b",
    r"\bdejeneratif\b", r"\bkikirdak kayb\w*\b",
    # de
    r"\barthrose\w*\b", r"\bgonarthros\w*\b", r"\bosteophyt\w*\b", r"\bknorpelverlust\w*\b",
    r"\bknorpelschaden\b", r"\bchondropath\w*\b", r"\bdegenerativ\w*\b",
    # nl
    r"\bartrose\w*\b", r"\bosteofyt\w*\b", r"\bkraakbeenverlies\b", r"\bchondropath\w*\b",
    r"\bdegeneratie\w*\b",
)


# ---------------------------------------------------------------------------
# The 12 findings
# ---------------------------------------------------------------------------
FINDING_SPECS: tuple[FindingSpec, ...] = (
    FindingSpec(
        column="ACL",
        self_sufficient=(
            r"\b(?:vorderes? )?kreuzband\w*(?:riss|ruptur)\w*\b",
            r"\bvkb[\s-]?(?:riss|ruptur)\w*\b",
            r"\b(?:voorste )?kruisband\w*(?:scheur|ruptuur)\w*\b",
        ),
        structures=(
            r"\bacl\b", r"\banterior cruciate ligament\b", r"\banterior cruciate\b",
            r"\blca\b", r"\bligamento cruzado anterior\b",
            r"\bon capraz bag\w*\b", r"\bocb\b", r"\bonn? capraz\b",
            r"\bvkb\b", r"\bvorderes? kreuzband\w*\b",
            r"\bvoorste kruisband\w*\b",
        ),
        pathology=TEAR_CUES,
        integrity=INTEGRITY_CUES,
    ),
    FindingSpec(
        column="MCL",
        self_sufficient=(
            r"\binnenband\w*(?:riss|ruptur)\w*\b",
            r"\bbinnenband\w*(?:scheur|ruptuur)\w*\b",
        ),
        structures=(
            r"\bmcl\b", r"\bmedial collateral ligament\b",
            r"\blcm\b", r"\bligamento colateral medial\b",
            r"\bligamento lateral interno\b", r"\blli\b",
            r"\bic yan bag\w*\b", r"\bmedial kollateral\w*\b",
            r"\binnenband\w*\b", r"\bmediales kollateralband\w*\b", r"\bmkb\b",
            r"\bmediale collaterale band\w*\b", r"\bbinnenband\w*\b",
        ),
        pathology=TEAR_CUES,
        integrity=INTEGRITY_CUES,
    ),
    FindingSpec(
        column="Medial Meniscus",
        # German and Dutch fuse the pathology into the noun
        # ("Innenmeniskusriss"), so no separate cue token exists to anchor to.
        self_sufficient=(
            r"\binnenmeniskus\w*(?:riss|ruptur)\w*\b",
            r"\bmedialer? meniskus\w*(?:riss|ruptur)\w*\b",
            r"\bmediale meniscus\w*(?:scheur|ruptuur)\w*\b",
            r"\bbinnenmeniscus\w*(?:scheur|ruptuur)\w*\b",
        ),
        structures=(
            r"\bmedial meniscus\b", r"\bmedial meniscal\b",
            r"\bmenisco interno\b", r"\bmenisco medial\b",
            r"\bic menisk\w*\b", r"\bmedial menisk\w*\b",
            r"\binnenmenisk\w*\b", r"\bmedialer? menisk\w*\b",
            r"\bmediale meniscus\b", r"\bbinnenmeniscus\b",
        ),
        pathology=TEAR_CUES,
        integrity=INTEGRITY_CUES,
    ),
    FindingSpec(
        column="Lateral Meniscus",
        self_sufficient=(
            r"\baussenmeniskus\w*(?:riss|ruptur)\w*\b",
            r"\blateraler? meniskus\w*(?:riss|ruptur)\w*\b",
            r"\blaterale meniscus\w*(?:scheur|ruptuur)\w*\b",
            r"\bbuitenmeniscus\w*(?:scheur|ruptuur)\w*\b",
        ),
        structures=(
            r"\blateral meniscus\b", r"\blateral meniscal\b",
            r"\bmenisco externo\b", r"\bmenisco lateral\b",
            r"\bdis menisk\w*\b", r"\blateral menisk\w*\b",
            r"\baussenmenisk\w*\b", r"\blateraler? menisk\w*\b",
            r"\blaterale meniscus\b", r"\bbuitenmeniscus\b",
        ),
        pathology=TEAR_CUES,
        integrity=INTEGRITY_CUES,
    ),
    FindingSpec(
        column="Medial OA",
        self_sufficient=(
            r"\bmediale?r? gonarthros\w*\b", r"\bgonartrosis medial\w*\b",
            r"\bmedial gonartroz\w*\b",
        ),
        structures=(
            r"\bmedial compartment\b", r"\bmedial femorotibial\b",
            r"\bcompartimento medial\b", r"\bcompartimento femorotibial interno\b",
            r"\bmedial kompartman\w*\b", r"\bmediales kompartiment\w*\b",
            r"\bmediale?n? femorotibial\w*\b", r"\bmediaal compartiment\b",
            r"\bmediale compartiment\w*\b",
        ),
        pathology=DEGENERATION_CUES,
    ),
    FindingSpec(
        column="Lateral OA",
        self_sufficient=(
            r"\blaterale?r? gonarthros\w*\b", r"\bgonartrosis lateral\w*\b",
            r"\blateral gonartroz\w*\b",
        ),
        structures=(
            r"\blateral compartment\b", r"\blateral femorotibial\b",
            r"\bcompartimento lateral\b", r"\bcompartimento femorotibial externo\b",
            r"\blateral kompartman\w*\b", r"\blaterales kompartiment\w*\b",
            r"\blaterale?n? femorotibial\w*\b", r"\blateraal compartiment\b",
            r"\blaterale compartiment\w*\b",
        ),
        pathology=DEGENERATION_CUES,
    ),
    FindingSpec(
        column="PF OA",
        self_sufficient=(
            r"\bpatellofemoral (?:osteo)?arthr\w*\b", r"\bpatellofemoral degenerativ\w*\b",
            r"\bchondromalacia patell\w*\b", r"\bcondromalacia rotulian\w*\b",
            r"\bartrosis femoropatelar\b", r"\bfemoropatelar degenerativ\w*\b",
            r"\bpatellofemoral artroz\w*\b", r"\bkondromalazi patella\w*\b",
            r"\bretropatellar\w* (?:arthrose|chondropath\w*)\b",
            r"\bfemoropatellar\w* arthros\w*\b", r"\bchondropathia patellae\b",
            r"\bpatellofemorale artrose\b",
        ),
        structures=(
            r"\bpatellofemoral\w*\b", r"\bfemoropatelar\w*\b", r"\bfemoropatellar\w*\b",
            r"\bretropatellar\w*\b", r"\bpatellofemorale?n?\b",
        ),
        pathology=DEGENERATION_CUES,
    ),
    FindingSpec(
        column="Effusion",
        self_sufficient=(
            r"\beffusions?\b", r"\bjoint fluid\b", r"\bintra ?articular fluid\b",
            r"\bderrame\w*\b",
            r"\befuzyon\w*\b", r"\beklem sivisi\b",
            r"\b\w*erguss\w*\b", r"\bgelenkflussigkeit\b",
            r"\bhydrops\b", r"\bgewrichtsvocht\b", r"\beffusie\b",
        ),
    ),
    FindingSpec(
        column="Synovitis",
        self_sufficient=(
            r"\bsynovit\w*\b", r"\bsynovial thickening\b", r"\bsynovial proliferat\w*\b",
            r"\bsinovit\w*\b", r"\bengrosamiento sinovial\b",
            r"\bsinovyal kalinlasma\b",
            r"\bsynovialit\w*\b", r"\bsynovialis verdick\w*\b",
            r"\bsynoviale verdikking\b",
        ),
    ),
    FindingSpec(
        column="Baker's",
        self_sufficient=(
            r"\bbakers? cyst\w*\b", r"\bbaker'?s cyst\w*\b", r"\bpopliteal cyst\w*\b",
            r"\bquiste de baker\b", r"\bquiste popliteo\b",
            r"\bbaker kist\w*\b", r"\bpopliteal kist\w*\b",
            r"\bbaker[\s-]?zyste\w*\b", r"\bbakerzyste\w*\b", r"\bpoplitealzyste\w*\b",
            r"\bbakercyste\w*\b", r"\bpopliteale cyste\w*\b",
        ),
    ),
    FindingSpec(
        column="Contusion",
        self_sufficient=(
            r"\bcontusion\w*\b", r"\bbone bruise\w*\b",
            r"\bbone marrow (?:o)?edema\b", r"\bmarrow (?:o)?edema\b",
            r"\bcontusion osea\b", r"\bedema oseo\b", r"\bedema de medula osea\b",
            r"\bkontuzyon\w*\b", r"\bkemik ilik odem\w*\b", r"\bkemik odem\w*\b",
            r"\bkontusion\w*\b", r"\bknochenmarkodem\w*\b", r"\bknochenodem\w*\b",
            r"\bcontusie\w*\b", r"\bbotcontusie\w*\b", r"\bbeenmergoedeem\b",
        ),
    ),
    FindingSpec(
        column="Fracture",
        self_sufficient=(
            r"\bfractur\w*\b",
            r"\bkirik\w*\b",
            r"\bfraktur\w*\b", r"\bknochenbruch\w*\b",
            r"\bfractu\w*\b", r"\bbotbreuk\w*\b",
        ),
    ),
)

SPEC_BY_COLUMN: dict[str, FindingSpec] = {s.column: s for s in FINDING_SPECS}
