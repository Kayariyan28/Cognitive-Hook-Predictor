"""Provenance and cache identity for one generated insight artifact.

Everything a reader needs to reproduce or distrust an artifact travels with it:
which provider ran, which model revision, which prompt template, what the
evidence hashed to, and what came back.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .bundle import canonical_json
from .config import InsightSettings
from .prompts.hook_doctor import PROMPT_TEMPLATE_ID, prompt_hash
from .providers.base import GenerationResult


PROVENANCE_SCHEMA_VERSION = "insight-provenance/1"

LIMITS_STATEMENT = (
    "This is a descriptive lane. Every sentence is generated from measured or "
    "model-derived evidence that is cited inline and checked against that evidence "
    "before publication. Hypotheses are untested heuristics, not findings. TRIBE "
    "values are predicted average-subject cortical BOLD, not audience behavior. "
    "Nothing here is a behavioral prediction, and no behavioral head is installed."
)


def output_hash(artifact: Mapping[str, Any]) -> str:
    """Digest the model-settable fields only; server-owned fields are not evidence."""

    settable = {
        key: artifact[key]
        for key in ("hookReport", "hookRewrites", "phaseCommentary", "tribeNotes")
        if key in artifact
    }
    return hashlib.sha256(canonical_json(settable).encode("utf-8")).hexdigest()


def build_provenance(
    *,
    settings: InsightSettings,
    generation: GenerationResult,
    input_evidence_hash: str,
    artifact: Mapping[str, Any],
    hook_only: bool,
) -> dict[str, Any]:
    return {
        "schemaVersion": PROVENANCE_SCHEMA_VERSION,
        "provider": settings.provider,
        "modelId": generation.model_id,
        "modelRevision": generation.model_revision,
        "promptTemplateId": PROMPT_TEMPLATE_ID,
        "promptHash": prompt_hash(),
        "temperature": settings.temperature,
        "maxOutputTokens": settings.max_output_tokens,
        "inputEvidenceHash": input_evidence_hash,
        "outputHash": output_hash(artifact),
        "hookOnly": hook_only,
        "startedAt": generation.started_at,
        "completedAt": generation.completed_at,
        "elapsedSeconds": generation.elapsed_seconds,
        "behavioralOutcome": False,
    }


def cache_key(
    *,
    input_evidence_hash: str,
    provider: str,
    model_revision: str,
    temperature: float,
) -> str:
    """An exact repeat of the same evidence, prompt, model, and temperature."""

    identity = canonical_json(
        {
            "inputEvidenceHash": input_evidence_hash,
            "promptHash": prompt_hash(),
            "provider": provider,
            "modelRevision": model_revision,
            "temperature": temperature,
        }
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
