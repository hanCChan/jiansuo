"""QA checks for XXX semantic fill results."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from xxx_utils import (
    EMAIL_RE,
    HINT_PII_NAME,
    LONG_DIGIT_RE,
    NUMERIC_PII_HINTS,
    find_xxx_tokens,
    strip_xxx,
)

GENERIC_PII_PHRASE_RE = re.compile(
    r"المعلومات المذكورة|رقم البطاقة المذكور|رقم الهاتف المسجل|البيانات المذكورة",
    re.I,
)

ENTITY_TOKENS = (
    "bca",
    "mybca",
    "flazz",
    "klikbca",
    "otp",
    "pin",
    "qris",
    "atm",
)

WARN_PRESERVATION_THRESHOLD = 0.72
FAIL_PRESERVATION_THRESHOLD = 0.55


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def preservation_ratio(original_msa: str, filled_msa: str) -> float:
    orig_xxx_count = len(find_xxx_tokens(original_msa))
    a = _normalize(strip_xxx(original_msa))
    b = _normalize(strip_xxx(filled_msa))
    if orig_xxx_count >= 3:
        # Mostly-XXX turns: compare skeleton only, ignoring synthetic digit fills.
        b = re.sub(r"\d{2,}", "", b)
        b = _normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def has_pii_shape(text: str) -> list[str]:
    issues: list[str] = []
    if LONG_DIGIT_RE.search(text):
        issues.append("long_digit_sequence")
    if EMAIL_RE.search(text):
        issues.append("email_shape")
    return issues


def missing_entities(original_msa: str, filled_msa: str) -> list[str]:
    low_orig = original_msa.lower()
    low_fill = filled_msa.lower()
    missing: list[str] = []
    for token in ENTITY_TOKENS:
        if token in low_orig and token not in low_fill:
            missing.append(token)
    return missing


def qa_turn_fill(
    original_msa: str,
    filled_msa: str,
    *,
    hint_type: str = "unknown",
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not filled_msa.strip():
        errors.append("empty_filled")

    orig_xxx = len(find_xxx_tokens(original_msa))
    left_xxx = len(find_xxx_tokens(filled_msa))
    if left_xxx > 0:
        errors.append(f"xxx_remaining:{left_xxx}/{orig_xxx}")

    ratio = preservation_ratio(original_msa, filled_msa)
    if ratio < FAIL_PRESERVATION_THRESHOLD:
        errors.append(f"non_xxx_preservation_low:{ratio:.3f}")
    elif ratio < WARN_PRESERVATION_THRESHOLD:
        warnings.append(f"non_xxx_preservation_warn:{ratio:.3f}")

    for issue in has_pii_shape(filled_msa):
        if issue == "email_shape":
            errors.append(f"pii_shape_{issue}")
        elif issue == "long_digit_sequence":
            if hint_type not in NUMERIC_PII_HINTS:
                errors.append(f"pii_shape_{issue}")

    if hint_type in NUMERIC_PII_HINTS:
        if GENERIC_PII_PHRASE_RE.search(filled_msa) and not LONG_DIGIT_RE.search(filled_msa):
            errors.append("pii_generic_phrase_instead_of_digits")
    elif hint_type == HINT_PII_NAME:
        if LONG_DIGIT_RE.search(filled_msa):
            errors.append("pii_name_has_long_digits")

    for entity in missing_entities(original_msa, filled_msa):
        errors.append(f"missing_entity:{entity}")

    status = "accepted"
    if errors:
        status = "failed"
    elif warnings:
        status = "warning"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "preservation_ratio": round(ratio, 4),
        "xxx_remaining": left_xxx,
    }


def qa_cluster_result(cluster_payload: dict[str, Any]) -> dict[str, Any]:
    """QA all turns returned for one cluster."""
    turn_results: dict[int, dict[str, Any]] = {}
    statuses: list[str] = []

    turns_by_num = {int(t["turn"]): t for t in cluster_payload.get("turns_to_fill", [])}
    for item in cluster_payload.get("turns", []):
        turn_num = int(item["turn"])
        src = turns_by_num.get(turn_num, {})
        original = src.get("content_msa", "")
        filled = item.get("content_msa_filled", "")
        hint = src.get("hint_type", "unknown")
        qa = qa_turn_fill(original, filled, hint_type=hint)
        turn_results[str(turn_num)] = qa
        statuses.append(qa["status"])

    if "failed" in statuses:
        cluster_status = "failed"
    elif "warning" in statuses:
        cluster_status = "warning"
    else:
        cluster_status = "accepted"

    return {
        "cluster_status": cluster_status,
        "turn_results": turn_results,
    }
