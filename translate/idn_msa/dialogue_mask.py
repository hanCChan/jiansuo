from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

SENSITIVE_TOKEN_RE = re.compile(r"\b[A-Z0-9]*X{2,}[A-Z0-9]*\b", re.IGNORECASE)
SENSITIVE_TOKEN_FULL_RE = re.compile(r"^[A-Z0-9]*X{2,}[A-Z0-9]*$", re.IGNORECASE)


def _is_sensitive_only_line(text: str) -> bool:
    """O(n) check for lines made only of XXX-style tokens (avoid regex backtracking)."""
    tokens = text.split()
    if not tokens:
        return False
    return all(SENSITIVE_TOKEN_FULL_RE.fullmatch(token) for token in tokens)


def load_skip_whitelist(config_dir: Path | None = None) -> dict[str, Any]:
    root = config_dir or Path(__file__).resolve().parent.parent / "config"
    path = root / "dialogue_skip_whitelist.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def should_skip_content(text: str, skip_cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Return (skip, reason). Skipped turns copy content -> content_msa."""
    raw = (text or "").strip()
    if not raw:
        return True, "empty_content"

    cfg = skip_cfg or load_skip_whitelist()
    exact = {str(x).strip().upper() for x in cfg.get("exact_skip", []) if str(x).strip()}
    if raw.upper() in exact:
        return True, "exact_skip"

    if _is_sensitive_only_line(raw):
        return True, "pattern_skip:sensitive_only_tokens"

    for pattern in cfg.get("pattern_skip", []):
        if re.fullmatch(pattern, raw, flags=re.IGNORECASE):
            return True, f"pattern_skip:{pattern}"

    return False, ""


def mask_sensitive_tokens(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Mask XXX / EXXX style sensitive spans before sending to Kimi."""
    pairs: list[tuple[str, str]] = []

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        placeholder = f"<SENS_{len(pairs):02d}>"
        pairs.append((placeholder, token))
        return placeholder

    masked = SENSITIVE_TOKEN_RE.sub(_repl, text)
    return masked, pairs


def restore_sensitive_tokens(text: str, pairs: list[tuple[str, str]]) -> str:
    restored = text
    for placeholder, original in pairs:
        restored = restored.replace(placeholder, original)
    return restored
