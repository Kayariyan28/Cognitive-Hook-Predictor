"""Isolated stdio runner for one NanoLLaVA six-keyframe batch.

This file is executed directly by a separately managed MLX-VLM Python
interpreter.  It intentionally imports no TRIBE backend modules.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any


REQUEST_SCHEMA = "forecast-nanollava-stdio-request/1"
RESPONSE_SCHEMA = "forecast-nanollava-stdio-response/1"
MODEL_REVISION = "6e2bc13b87ab178668313552b9d69026af7d556f"
MODEL_WEIGHTS_SHA256 = (
    "f0a1d1517ad9e6e810bc6ac99956643c66e4b87a2f82bd5d1b5cb0966e5c5476"
)
KEYFRAME_COUNT = 6
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_CHARACTERS = 12_000
MAX_TOKENS = 320
OFFLINE_ENVIRONMENT = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)

SEMANTIC_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "scene",
        "action",
        "visibleText",
        "shot",
        "uncertainties",
    ],
    "properties": {
        "scene": {"type": "string", "minLength": 1, "maxLength": 240},
        "action": {"type": "string", "minLength": 1, "maxLength": 180},
        # NanoLLaVA is not accepted as an OCR provider. JSON null records
        # unavailable evidence and cannot be mistaken for observed no-text.
        "visibleText": {"type": "null"},
        "shot": {"type": "string", "minLength": 1, "maxLength": 160},
        "uncertainties": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 160,
                    },
                },
            ]
        },
    },
}


class StdioContractError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_REQUEST_BYTES:
        raise StdioContractError("request size is invalid")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StdioContractError("request contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StdioContractError("request is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise StdioContractError("request must be an object")
    return value


def _validate_request(value: dict[str, Any]) -> tuple[Path, str, list[Path], int]:
    if set(value) != {
        "schemaVersion",
        "snapshot",
        "modelRevision",
        "modelWeightsSha256",
        "prompt",
        "maxTokens",
        "frames",
    }:
        raise StdioContractError("request fields are invalid")
    if value["schemaVersion"] != REQUEST_SCHEMA:
        raise StdioContractError("request schema is unsupported")
    if value["modelRevision"] != MODEL_REVISION:
        raise StdioContractError("model revision is not pinned")
    if value["modelWeightsSha256"] != MODEL_WEIGHTS_SHA256:
        raise StdioContractError("model hash is not pinned")
    if value["maxTokens"] != MAX_TOKENS:
        raise StdioContractError("generation bound is invalid")
    prompt = value["prompt"]
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or prompt != prompt.strip()
        or len(prompt) > 8192
    ):
        raise StdioContractError("prompt is invalid")
    raw_snapshot = value["snapshot"]
    if not isinstance(raw_snapshot, str) or not Path(raw_snapshot).is_absolute():
        raise StdioContractError("snapshot must be absolute")
    snapshot = Path(raw_snapshot)
    if not snapshot.is_dir():
        raise StdioContractError("snapshot is unavailable")
    weights = snapshot / "model.safetensors"
    if not weights.is_file() or _sha256_file(weights) != MODEL_WEIGHTS_SHA256:
        raise StdioContractError("snapshot weights failed verification")

    raw_frames = value["frames"]
    if not isinstance(raw_frames, list) or len(raw_frames) != KEYFRAME_COUNT:
        raise StdioContractError("exactly six frames are required")
    paths: list[Path] = []
    for expected_index, item in enumerate(raw_frames):
        if not isinstance(item, dict) or set(item) != {"index", "path", "sha256"}:
            raise StdioContractError("frame fields are invalid")
        if item["index"] != expected_index:
            raise StdioContractError("frame order is invalid")
        raw_path = item["path"]
        digest = item["sha256"]
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise StdioContractError("frame path must be absolute")
        path = Path(raw_path)
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or len(digest) != 64
            or _sha256_file(path) != digest
        ):
            raise StdioContractError("frame failed verification")
        paths.append(path)
    return snapshot, prompt, paths, MAX_TOKENS


def _generate_one(
    *,
    model: Any,
    processor: Any,
    generate: Any,
    apply_chat_template: Any,
    logits_processor: Any,
    image_path: Path,
    prompt: str,
    max_tokens: int,
) -> str:
    formatted_prompt = f"<image>\n{prompt}"
    if callable(apply_chat_template):
        formatted_prompt = apply_chat_template(
            processor,
            getattr(model, "config", {}),
            prompt,
            add_generation_prompt=True,
            num_images=1,
        )
    if not isinstance(formatted_prompt, str) or not formatted_prompt:
        raise StdioContractError("chat template returned an invalid prompt")
    kwargs: dict[str, Any] = {"verbose": False, "max_tokens": max_tokens}
    kwargs["logits_processors"] = [logits_processor]
    try:
        parameters = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "temperature" in parameters:
        kwargs["temperature"] = 0.0
    if "image" in parameters:
        kwargs["image"] = str(image_path)
        result = generate(model, processor, formatted_prompt, **kwargs)
    elif "images" in parameters:
        kwargs["images"] = [str(image_path)]
        result = generate(model, processor, formatted_prompt, **kwargs)
    else:
        result = generate(
            model,
            processor,
            formatted_prompt,
            str(image_path),
            **kwargs,
        )
    text = result if isinstance(result, str) else getattr(result, "text", None)
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > MAX_RESPONSE_CHARACTERS
    ):
        raise StdioContractError("generation returned invalid text")
    return text.strip()


def _execute(value: dict[str, Any]) -> dict[str, Any]:
    if any(os.environ.get(name) != "1" for name in OFFLINE_ENVIRONMENT):
        raise StdioContractError("offline execution flags are required")
    snapshot, prompt, frames, max_tokens = _validate_request(value)
    try:
        from mlx_vlm import apply_chat_template, generate, load, structured
    except ImportError as exc:
        raise StdioContractError("mlx_vlm is unavailable") from exc

    # Some model/runtime versions print progress to stdout. Redirect it to
    # stderr so stdout remains one strict machine-readable JSON document.
    with contextlib.redirect_stdout(sys.stderr):
        # The pinned checkpoint works through MLX-VLM's built-in llava-qwen2
        # remapping. Never execute the snapshot's unpinned custom Python files.
        model, processor = load(
            str(snapshot),
            strict=True,
            trust_remote_code=False,
        )
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        schema_processor = structured.build_json_schema_logits_processor(
            tokenizer,
            SEMANTIC_JSON_SCHEMA,
        )
        responses = [
            _generate_one(
                model=model,
                processor=processor,
                generate=generate,
                apply_chat_template=apply_chat_template,
                logits_processor=schema_processor.clone(),
                image_path=path,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            for path in frames
        ]
    return {
        "schemaVersion": RESPONSE_SCHEMA,
        "responses": [
            {
                "index": index,
                "frameSha256": _sha256_file(frames[index]),
                "text": text,
            }
            for index, text in enumerate(responses)
        ],
    }


def main() -> int:
    try:
        payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        output = _execute(_strict_object(payload))
    except Exception:
        sys.stderr.write(
            '{"error":{"code":"nanollava_isolated_runtime_failed"}}\n'
        )
        return 2
    sys.stdout.write(
        json.dumps(
            output,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
