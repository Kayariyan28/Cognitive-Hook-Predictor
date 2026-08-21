"""An optional LLM tripwire for phrasing the term lists do not catch.

This is the third enforcement layer's last resort, and it is deliberately the
weakest one. It never runs in the request path, it never decides whether an
artifact may be published, and it is skipped entirely unless an operator
enables it and configures a provider. Its only job is to notice a violation the
deterministic validator's vocabulary missed, so the vocabulary can be extended.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping

from .claim_terms import (
    GLOBAL_MENTAL_STATE_TERMS,
    GLOBAL_OUTCOME_TERMS,
    TRIBE_SCOPED_TERMS,
    prompt_term_block,
)


JUDGE_PROMPT_ID = "insight-judge.v1"

JUDGE_SYSTEM_PROMPT = f"""\
You are auditing generated text from a video-analysis tool. The tool measures clips. It
never observes an audience, and it has no behavioral model of any kind.

You will receive a JSON array of items. Each item has an `itemPath`, its `text`, and
whether it cites predicted cortical model output (`citesTribe`).

Report every place the text does one of the following:

1. Asserts or implies an audience or platform outcome — for example: \
{prompt_term_block(GLOBAL_OUTCOME_TERMS)} — or any paraphrase of one.
2. Attributes a mental state to a viewer — for example: \
{prompt_term_block(GLOBAL_MENTAL_STATE_TERMS)} — or any paraphrase of one.
3. In an item where `citesTribe` is true, reverse-infers from cortical values — for
example: {prompt_term_block(TRIBE_SCOPED_TERMS)} — or describes those values as anything
other than predicted average-subject cortical BOLD.

A sentence that names one of these only to deny it is not a violation.

Return exactly one JSON object and nothing else:

{{"violations": [{{"itemPath": "...", "quote": "...", "reason": "..."}}]}}

Return {{"violations": []}} when the text is clean."""


class JudgeUnavailable(RuntimeError):
    """No provider is configured, so the tripwire cannot run."""


def collect_items(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten one artifact's model-authored text into judgeable items."""

    items: list[dict[str, Any]] = []

    def add(path: str, text: Any, citations: Any) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        cites = [item for item in (citations or ()) if isinstance(item, str)]
        items.append(
            {
                "itemPath": path,
                "text": text,
                "citesTribe": any(item.startswith("tribe:") for item in cites),
            }
        )

    hook_report = artifact.get("hookReport")
    if isinstance(hook_report, Mapping):
        for name in ("whatTheHookContains", "observations", "hypotheses"):
            for index, entry in enumerate(hook_report.get(name) or ()):
                if isinstance(entry, Mapping):
                    add(f"/hookReport/{name}/{index}/text", entry.get("text"), entry.get("citations"))
        for index, entry in enumerate(hook_report.get("experiments") or ()):
            if isinstance(entry, Mapping):
                add(
                    f"/hookReport/experiments/{index}/edit",
                    entry.get("edit"),
                    entry.get("citations"),
                )
    for index, entry in enumerate(artifact.get("hookRewrites") or ()):
        if isinstance(entry, Mapping):
            add(f"/hookRewrites/{index}/line", entry.get("line"), entry.get("citations"))
    for name in ("phaseCommentary", "tribeNotes"):
        for index, entry in enumerate(artifact.get(name) or ()):
            if isinstance(entry, Mapping):
                add(f"/{name}/{index}/text", entry.get("text"), entry.get("citations"))
    return items


def build_judge_request(artifacts: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    """Build the judge's system and user messages from validated artifacts."""

    items: list[dict[str, Any]] = []
    for artifact in artifacts:
        items.extend(collect_items(artifact))
    if not items:
        raise JudgeUnavailable("there is no generated text to audit")
    return JUDGE_SYSTEM_PROMPT, json.dumps(items, ensure_ascii=False, sort_keys=True)


def parse_judge_response(raw: str) -> list[dict[str, Any]]:
    """Read the judge strictly; an unreadable verdict is not a clean verdict."""

    try:
        payload = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise JudgeUnavailable(f"the judge returned unreadable JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("violations"), list):
        raise JudgeUnavailable("the judge returned no violations array")
    violations: list[dict[str, Any]] = []
    for entry in payload["violations"]:
        if not isinstance(entry, Mapping):
            raise JudgeUnavailable("the judge returned a malformed violation")
        violations.append(
            {
                "itemPath": entry.get("itemPath"),
                "quote": entry.get("quote"),
                "reason": entry.get("reason"),
            }
        )
    return violations


def judge_artifacts(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    generate: Callable[[str, str], str],
) -> list[dict[str, Any]]:
    """Run the tripwire. `generate` is any (system, user) -> text callable."""

    system, user = build_judge_request(artifacts)
    return parse_judge_response(generate(system, user))
