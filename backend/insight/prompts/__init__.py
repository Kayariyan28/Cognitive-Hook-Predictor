"""Versioned prompt templates. A template's hash travels in every manifest."""

from .hook_doctor_v1 import (
    PROMPT_TEMPLATE_ID,
    build_user_message,
    prompt_hash,
    system_prompt,
)

__all__ = ["PROMPT_TEMPLATE_ID", "build_user_message", "prompt_hash", "system_prompt"]
