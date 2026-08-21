"""Descriptive insight layer over already-validated SignalFrame evidence.

Nothing in this package measures anything, loads media, or emits a behavioral
value.  It assembles validated evidence into a citable JSON bundle, asks a
configured language model for text about that bundle, and refuses to publish
anything the bundle does not support.
"""

from .bundle import (
    BUNDLE_SCHEMA_VERSION,
    HOOK_WINDOW_SECONDS,
    LANE_KEYS,
    BundleUnavailableError,
    absent_lane,
    assemble_evidence_bundle,
    canonical_json,
    hook_evidence_card,
    input_evidence_hash,
)

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "HOOK_WINDOW_SECONDS",
    "LANE_KEYS",
    "BundleUnavailableError",
    "absent_lane",
    "assemble_evidence_bundle",
    "canonical_json",
    "hook_evidence_card",
    "input_evidence_hash",
]
