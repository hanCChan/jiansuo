from __future__ import annotations

import re
from typing import Any

from .config_loader import load_triage_config

ENT_PATTERN = re.compile(r"<ENT[^>]*>", re.IGNORECASE)
KIMI_PLACEHOLDER_RE = ENT_PATTERN
HOLLOW_PATTERN = re.compile(r"بطاقة\s+[؟،]|[؟]\s*$|سعر\s+إصدار|الحصول على بطاقة\s+؟")


def _error_types(hard_errors: list[str]) -> list[str]:
    return [e.split(":")[0] for e in hard_errors]


def _missing_entities(hard_errors: list[str]) -> list[str]:
    return [e.split(":", 1)[1] for e in hard_errors if e.startswith("missing_entity:")]


def _has_easy_theme(text: str, triage_cfg: dict[str, Any]) -> bool:
    themes = triage_cfg.get("easy_negative_themes", [])
    return any(theme.lower() in text.lower() for theme in themes)


def _is_hollow_translation(msa: str) -> bool:
    if ENT_PATTERN.search(msa):
        return True
    return bool(HOLLOW_PATTERN.search(msa))


def _source_has_dual_polarity(source_idn: str) -> bool:
    lower = source_idn.lower()
    blocked = any(p in lower for p in ("terblokir", "diblokir", "blokir"))
    unblocked = any(p in lower for p in ("membuka blokir", "buka blokir"))
    return blocked and unblocked


def _only_translatable_missing(hard_errors: list[str], triage_cfg: dict[str, Any]) -> bool:
    missing = _missing_entities(hard_errors)
    if not missing:
        return False
    translatable = set(triage_cfg.get("translatable_entities", []))
    other_errors = [e for e in _error_types(hard_errors) if e != "missing_entity"]
    return not other_errors and all(m in translatable for m in missing)


def _core_entity_missing(hard_errors: list[str], triage_cfg: dict[str, Any]) -> bool:
    core = set(triage_cfg.get("core_must_preserve", []))
    for entity in _missing_entities(hard_errors):
        if entity in core or any(c.lower() in entity.lower() for c in core):
            return True
    return False


def triage_failed_item(row: dict[str, Any], triage_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    triage_cfg = triage_cfg or load_triage_config()
    hard_errors = row.get("hard_errors", [])
    error_types = _error_types(hard_errors)
    role = row.get("role", "negative")
    source = row.get("source_idn", "")
    msa = row.get("msa", "")

    priority = "P1"
    action = "repair_translation"
    reason = "default negative repair"

    if role in {"query", "positive"}:
        return {
            **row,
            "repair_priority": "P0",
            "repair_action": "repair_translation",
            "reason": f"{role} must pass for eval",
        }

    if _only_translatable_missing(hard_errors, triage_cfg):
        return {
            **row,
            "repair_priority": "RULE",
            "repair_action": "adjust_qa_rule",
            "reason": "translatable entity reasonably localized to Arabic",
        }

    if "action_polarity_error" in error_types and _source_has_dual_polarity(source):
        return {
            **row,
            "repair_priority": "RULE",
            "repair_action": "adjust_qa_rule",
            "reason": "source describes both blocked state and unblock action",
        }

    if "term_confusion" in error_types or any(
        e.startswith("must_translate_latin:") for e in hard_errors
    ):
        return {
            **row,
            "repair_priority": "P0",
            "repair_action": "repair_translation",
            "reason": "critical term confusion (PIN/password/OTP boundary)",
        }

    if "unrestored_entity_placeholder" in error_types or KIMI_PLACEHOLDER_RE.search(msa):
        return {
            **row,
            "repair_priority": "P0",
            "repair_action": "repair_translation",
            "reason": "placeholder or product name lost in translation",
        }

    if _core_entity_missing(hard_errors, triage_cfg):
        return {
            **row,
            "repair_priority": "P0",
            "repair_action": "repair_translation",
            "reason": "core BCA/product entity missing",
        }

    if _is_hollow_translation(msa):
        return {
            **row,
            "repair_priority": "P0",
            "repair_action": "repair_translation",
            "reason": "translation hollow or product name dropped",
        }

    if _has_easy_theme(source, triage_cfg) and role == "negative":
        if _is_hollow_translation(msa) or "arabic_ratio_too_low" in error_types:
            return {
                **row,
                "repair_priority": "P2_DROP",
                "repair_action": "drop",
                "reason": "low-value themed negative; partial eval has enough negatives",
            }

    if "missing_entity" in error_types:
        return {
            **row,
            "repair_priority": "P1",
            "repair_action": "repair_translation",
            "reason": "product/service entity missing but recoverable",
        }

    if "action_polarity_error" in error_types:
        return {
            **row,
            "repair_priority": "P1",
            "repair_action": "repair_translation",
            "reason": "action polarity may need repair after rule pass",
        }

    if "arabic_ratio_too_low" in error_types:
        return {
            **row,
            "repair_priority": "P2_DROP",
            "repair_action": "drop",
            "reason": "ratio-only failure on low-value negative",
        }

    return {**row, "repair_priority": priority, "repair_action": action, "reason": reason}
