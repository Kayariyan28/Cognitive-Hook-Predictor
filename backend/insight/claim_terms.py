"""The single source of truth for insight claim-boundary vocabulary.

The prompt template embeds these lists, the deterministic validator enforces
them, and the frontend static check reads them through ``export_json``.  If a
term appears anywhere else in the repository as a literal list, that copy is the
defect: delete it and import from here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


CLAIM_TERMS_SCHEMA_VERSION = "insight-claim-terms/1"

# Platform or audience outcome claims. Forbidden in every insight string.
GLOBAL_OUTCOME_TERMS: tuple[str, ...] = (
    "viral",
    "virality",
    "go viral",
    "views",
    "view count",
    "watch time",
    "retention",
    "completion rate",
    "engagement rate",
    "click-through",
    "ctr",
    "impressions",
    "reach",
    "shares",
    "share rate",
    "followers",
    "subscribers",
    "algorithm",
    "the algorithm",
    "conversion",
    "converts",
    "monetization",
    "sells",
    "will perform",
    "outperform",
    # Found by the comparative red-team fixture: a rank restated as a verdict.
    "performance",
    "performing",
    "performs",
    "best-performing",
    "top-performing",
    "worst-performing",
    "guarantee",
    "guaranteed",
    "boost",
    "drive traffic",
)

# Attributing a mental state to a viewer. Forbidden in every insight string.
GLOBAL_MENTAL_STATE_TERMS: tuple[str, ...] = (
    "attention",
    "attentive",
    "engaged",
    "engagement",
    "emotion",
    "emotional",
    "feels",
    "feel",
    "felt",
    "mood",
    "arousal",
    "memory",
    "memorable",
    "remember",
    "bored",
    "boredom",
    "curiosity",
    "curious",
    "excitement",
    "excited",
    "interest",
    "interested",
    "hooked",
    "captivated",
    "empathy",
    "trust",
    "desire",
    "intent",
)

# Reverse inference from cortical values. Forbidden only where tribe: is cited.
TRIBE_SCOPED_TERMS: tuple[str, ...] = (
    "brain activity",
    "brain lights up",
    "lights up",
    "neural attention",
    "neuroscience proves",
    "the brain says",
    "subconscious",
    "dopamine",
    "limbic",
    "amygdala",
    "viewers' brains",
    "measured brain",
    "real brain",
    "live brain",
    "mind reading",
    "what viewers think",
)

# A sentence carrying one of these is exempt outright: it is already stating a
# limit rather than making a claim.
WHOLE_SENTENCE_LIMITERS: tuple[str, ...] = (
    "untested heuristic",
    "not audience behavior",
    "not a prediction",
    "not a measurement",
    "not a measure of",
    "no evidence",
    "predicted average-subject cortical bold",
    "predicted cortical bold",
    "not measured",
    "does not establish",
    "cannot establish",
    "not an outcome",
)

# A negation this close to the matched term exempts that one match.
PROXIMATE_NEGATIONS: tuple[str, ...] = (
    "not",
    "never",
    "no",
    "cannot",
    "isn't",
    "aren't",
    "doesn't",
    "don't",
    "without",
)
PROXIMITY_CHARACTERS = 60

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _term_pattern(term: str) -> re.Pattern[str]:
    """Match a term or phrase on whole-word boundaries, case-insensitively.

    Word boundaries keep ``hook`` and ``hooks`` legal while ``hooked`` is not,
    which is the difference between naming the product surface and asserting a
    viewer's mental state.
    """

    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


_GLOBAL_TERMS: tuple[str, ...] = GLOBAL_OUTCOME_TERMS + GLOBAL_MENTAL_STATE_TERMS
# Longest first so "go viral" is reported instead of the shorter "viral".
GLOBAL_TERMS: tuple[str, ...] = tuple(sorted(set(_GLOBAL_TERMS), key=lambda term: (-len(term), term)))
TRIBE_TERMS: tuple[str, ...] = tuple(
    sorted(set(TRIBE_SCOPED_TERMS), key=lambda term: (-len(term), term))
)

_GLOBAL_PATTERNS = tuple((term, _term_pattern(term)) for term in GLOBAL_TERMS)
_TRIBE_PATTERNS = tuple((term, _term_pattern(term)) for term in TRIBE_TERMS)
_NEGATION_PATTERNS = tuple(_term_pattern(word) for word in PROXIMATE_NEGATIONS)


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation and newlines; violations are sentence-scoped."""

    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


def _has_whole_sentence_limiter(lowered: str) -> bool:
    return any(limiter in lowered for limiter in WHOLE_SENTENCE_LIMITERS)


def _negation_is_proximate(sentence: str, start: int, end: int) -> bool:
    for pattern in _NEGATION_PATTERNS:
        for match in pattern.finditer(sentence):
            if match.start() >= end:
                distance = match.start() - end
            elif match.end() <= start:
                distance = start - match.end()
            else:
                distance = 0
            if distance <= PROXIMITY_CHARACTERS:
                return True
    return False


def sentence_violations(sentence: str, *, tribe_scoped: bool) -> list[str]:
    """Return the forbidden terms this sentence asserts without a limit."""

    lowered = sentence.lower()
    if _has_whole_sentence_limiter(lowered):
        return []
    patterns = _GLOBAL_PATTERNS + (_TRIBE_PATTERNS if tribe_scoped else ())
    found: list[str] = []
    consumed: list[tuple[int, int]] = []
    for term, pattern in patterns:
        for match in pattern.finditer(sentence):
            span = (match.start(), match.end())
            # A longer phrase already reported covers its shorter substrings.
            if any(start <= span[0] and span[1] <= end for start, end in consumed):
                continue
            if _negation_is_proximate(sentence, *span):
                continue
            consumed.append(span)
            found.append(term)
    return found


def export_json() -> str:
    """Serialize the vocabulary for consumers outside Python (the frontend check)."""

    return json.dumps(
        {
            "schemaVersion": CLAIM_TERMS_SCHEMA_VERSION,
            "globalOutcomeTerms": list(GLOBAL_OUTCOME_TERMS),
            "globalMentalStateTerms": list(GLOBAL_MENTAL_STATE_TERMS),
            "tribeScopedTerms": list(TRIBE_SCOPED_TERMS),
            "wholeSentenceLimiters": list(WHOLE_SENTENCE_LIMITERS),
            "proximateNegations": list(PROXIMATE_NEGATIONS),
            "proximityCharacters": PROXIMITY_CHARACTERS,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def prompt_term_block(terms: Iterable[str]) -> str:
    """Render one term list for embedding in the versioned prompt template."""

    return ", ".join(sorted(set(terms)))


def as_dict() -> dict[str, Any]:
    return json.loads(export_json())
