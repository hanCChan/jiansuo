from __future__ import annotations

import json
import re
from typing import Any

from .config_loader import PipelineConfig
from .expand import TranslationItem
from .unicode_norm import normalize_for_embedding, normalize_unicode

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
LATIN_RE = re.compile(r"[A-Za-z]+")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
PLACEHOLDER_RE = re.compile(r"<ENT_\d{2}>")


def _latin_tokens(text: str) -> list[str]:
    return LATIN_RE.findall(text)


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic_chars = len(ARABIC_RE.findall(text))
    return arabic_chars / max(len(text), 1)


def check_structure(items: list[TranslationItem], expected_counts: dict[str, int]) -> list[str]:
    errors: list[str] = []
    roles = {"query": 0, "positive": 0, "negative": 0}
    for item in items:
        roles[item.role] += 1
    for role, count in expected_counts.items():
        if roles.get(role, 0) != count:
            errors.append(f"count mismatch for {role}: expected {count}, got {roles.get(role, 0)}")
    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        errors.append("duplicate item_id detected")
    return errors


def hard_qa_item(item: TranslationItem, cfg: PipelineConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    text = item.msa_clean or item.msa_raw or ""

    if not text.strip():
        errors.append("empty_translation")

    if PLACEHOLDER_RE.search(text):
        errors.append("unrestored_entity_placeholder")

    if CJK_RE.search(text):
        errors.append("cjk_characters_found")

    ratio = _arabic_ratio(text)
    if ratio < 0.55:
        errors.append(f"arabic_ratio_too_low:{ratio:.3f}")

    whitelist = {w.lower() for w in cfg.latin_whitelist}
    residue = {w.lower() for w in cfg.residue_words}
    allowed_fragments = {"face", "id", "plus", "bisnis", "individu", "corporate"}
    for token in _latin_tokens(text):
        low = token.lower()
        if low in whitelist or low in allowed_fragments:
            continue
        if any(low in w.lower() or w.lower() in low for w in cfg.latin_whitelist):
            continue
        if low in residue:
            errors.append(f"indonesian_residue:{token}")
        else:
            errors.append(f"latin_leakage:{token}")

    for entity in item.entities_found:
        if entity not in text and entity.lower() not in text.lower():
            errors.append(f"missing_entity:{entity}")

    entity_conf = cfg.confusions.get("entity_confusions", {})
    for src_entity, forbidden_list in entity_conf.items():
        if src_entity in item.entities_found:
            for forbidden in forbidden_list:
                if forbidden in text and forbidden != src_entity:
                    errors.append(f"entity_drift:{src_entity}->{forbidden}")

    lower_src = item.source_idn.lower()
    lower_tgt = text.lower()
    for term_name, spec in cfg.glossary.get("terms", {}).items():
        triggered = any(t.lower() in lower_src for t in spec.get("id_triggers", []))
        if not triggered:
            continue
        expected = [e.lower() for e in spec.get("ar_expected", [])]
        forbidden = [f.lower() for f in spec.get("ar_forbidden", [])]
        if expected and not any(e in lower_tgt for e in expected):
            warnings.append(f"term_missing:{term_name}")
        if any(f in lower_tgt for f in forbidden):
            errors.append(f"term_confusion:{term_name}")

    for action_name, spec in cfg.action_polarity.get("actions", {}).items():
        if action_name not in item.actions_found:
            continue
        expected = [e.lower() for e in spec.get("ar_expected", [])]
        forbidden = [f.lower() for f in spec.get("ar_forbidden", [])]
        if expected and not any(e in lower_tgt for e in expected):
            warnings.append(f"action_missing:{action_name}")
        if forbidden and any(f in lower_tgt for f in forbidden):
            errors.append(f"action_polarity_error:{action_name}")

    src_len = max(len(item.source_idn), 1)
    tgt_len = len(text)
    len_ratio = tgt_len / src_len
    if len_ratio < 0.35:
        warnings.append(f"maybe_too_short:{len_ratio:.2f}")
    if len_ratio > 3.5:
        warnings.append(f"maybe_too_long:{len_ratio:.2f}")

    if "```" in text or text.strip().startswith("{"):
        errors.append("markdown_or_json_leakage")

    hard_pass = len(errors) == 0
    return {
        "hard_pass": hard_pass,
        "hard_errors": errors,
        "hard_warnings": warnings,
        "arabic_ratio": round(ratio, 4),
        "latin_tokens": _latin_tokens(text),
    }


def apply_postprocess(item: TranslationItem, cfg: PipelineConfig) -> None:
    item.msa_clean = normalize_unicode(item.msa_raw)
    item.msa_norm_for_embedding = normalize_for_embedding(item.msa_raw)


def validate_json_payload(payload: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(payload)
    except json.JSONDecodeError as exc:
        return False, str(exc)
