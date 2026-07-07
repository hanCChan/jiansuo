from __future__ import annotations

import json
import re
from typing import Any

from .config_loader import PipelineConfig, load_triage_config
from .expand import TranslationItem
from .unicode_norm import normalize_for_embedding, normalize_unicode

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
ARABIC_WORD_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
)
LATIN_RE = re.compile(r"[A-Za-z]+")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
PLACEHOLDER_RE = re.compile(r"<ENT_\d{2}>")
KIMI_PLACEHOLDER_RE = re.compile(r"<ENT[^>]*>", re.IGNORECASE)


def _latin_tokens(text: str) -> list[str]:
    return LATIN_RE.findall(text)


def _source_latin_allowlist(item: TranslationItem) -> set[str]:
    allowed: set[str] = set()
    for token in _latin_tokens(item.source_idn):
        allowed.add(token)
        allowed.add(token.lower())
    for entity in item.entities_found:
        for token in _latin_tokens(entity):
            allowed.add(token)
            allowed.add(token.lower())
    return allowed


def _must_translate_latin(cfg: PipelineConfig) -> set[str]:
    must: set[str] = {w.lower() for w in cfg.residue_words}
    for term_name, spec in cfg.glossary.get("terms", {}).items():
        if term_name in {"pin", "otp"}:
            continue
        for trigger in spec.get("id_triggers", []):
            if trigger.isascii():
                must.add(trigger.lower())
    return must


def _is_whitelisted_latin(
    token: str,
    cfg: PipelineConfig,
    source_allowlist: set[str] | None = None,
) -> bool:
    low = token.lower()
    if source_allowlist and (token in source_allowlist or low in source_allowlist):
        return True
    whitelist = {w.lower() for w in cfg.latin_whitelist}
    allowed_fragments = {"face", "id", "plus", "bisnis", "individu", "corporate", "online", "card"}
    if low in whitelist or low in allowed_fragments:
        return True
    return any(low in w.lower() or w.lower() in low for w in cfg.latin_whitelist)


def _arabic_ratio(text: str, cfg: PipelineConfig, source_allowlist: set[str]) -> float:
    if not text:
        return 0.0
    arabic_chars = len(ARABIC_RE.findall(text))
    non_arabic = "".join(ch for ch in text if not ARABIC_RE.match(ch))
    latin_tokens = _latin_tokens(non_arabic)
    whitelisted_latin_chars = sum(
        len(tok) for tok in latin_tokens if _is_whitelisted_latin(tok, cfg, source_allowlist)
    )
    denom = max(len(text) - whitelisted_latin_chars, 1)
    return arabic_chars / denom


def _contains_forbidden_phrase(text: str, forbidden: str) -> bool:
    if forbidden not in text:
        return False
    if " " in forbidden:
        return forbidden in text
    unblock_markers = ("فك", "إلغاء", "إعادة", "رفع")
    unblock_phrases = ("فك الحظر", "فك حظر", "إلغاء الحظر", "إعادة فتح", "رفع الحظر")
    for phrase in unblock_phrases:
        if forbidden in phrase and phrase in text:
            return False
    if forbidden in ("حظر", "إيقاف", "تجميد"):
        for marker in unblock_markers:
            if marker in text and forbidden in text:
                return False
    if forbidden in ("تفعيل", "تنشيط"):
        for phrase in ("إلغاء تفعيل", "إيقاف التفعيل"):
            if phrase in text:
                return False
    return True


def _entities_present_in_text(text: str, entities: list[str]) -> list[str]:
    lower_text = text.lower()
    found: list[str] = []
    for entity in sorted(entities, key=len, reverse=True):
        if entity in text or entity.lower() in lower_text:
            found.append(entity)
    return found


def _is_entity_drift(
    src_entity: str,
    forbidden: str,
    text: str,
    source_entities: set[str],
    source_text: str,
    all_entities: list[str],
) -> bool:
    if forbidden not in text and forbidden.lower() not in text.lower():
        return False
    if forbidden in source_entities or forbidden in source_text:
        return False
    if src_entity in text or src_entity.lower() in text.lower():
        return False
    text_entities = _entities_present_in_text(text, all_entities)
    for entity in text_entities:
        if forbidden in entity and entity in source_entities:
            return False
    return True


def _entity_satisfied_by_translation(entity: str, text: str, cfg: PipelineConfig) -> bool:
    triage = load_triage_config(cfg.config_dir)
    translatable = set(triage.get("translatable_entities", []))
    if entity not in translatable:
        return False
    aliases = triage.get("translatable_ar_aliases", {}).get(entity, [])
    lower_text = text.lower()
    return any(alias.lower() in lower_text for alias in aliases)


def _dual_block_unblock_source(item: TranslationItem) -> bool:
    lower = item.source_idn.lower()
    blocked = any(p in lower for p in ("terblokir", "terblok", "diblokir", "blokir"))
    unblocked = any(p in lower for p in ("membuka blokir", "buka blokir", "pembukaan blokir"))
    return blocked and unblocked


def _has_unblock_phrase(text: str) -> bool:
    phrases = ("فك الحظر", "فك حظر", "إلغاء الحظر", "إعادة فتح", "رفع الحظر")
    return any(p in text for p in phrases)


def _dual_activate_deactivate_source(item: TranslationItem) -> bool:
    lower = item.source_idn.lower()
    activated = any(p in lower for p in ("mengaktifkan", "aktivasi"))
    deactivated = any(p in lower for p in ("menonaktifkan", "ditutup", "nonaktif"))
    return activated and deactivated


def _has_activate_phrase(text: str) -> bool:
    norm = normalize_for_embedding(text)
    phrases = ("تفعيل", "تنشيط", "إعادة تنشيط", "إعادة تفعيل")
    return any(p in norm for p in phrases)


def _has_deactivate_phrase(text: str) -> bool:
    norm = normalize_for_embedding(text)
    phrases = ("إيقاف", "إغلاق", "أغلق", "تعطيل", "إلغاء تفعيل")
    return any(p in norm for p in phrases)


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
    source_allowlist = _source_latin_allowlist(item)
    must_translate = _must_translate_latin(cfg)

    if not text.strip():
        errors.append("empty_translation")

    if PLACEHOLDER_RE.search(text):
        errors.append("unrestored_entity_placeholder")
    elif KIMI_PLACEHOLDER_RE.search(text):
        errors.append("kimi_hallucinated_placeholder")

    if CJK_RE.search(text):
        errors.append("cjk_characters_found")

    ratio = _arabic_ratio(text, cfg, source_allowlist)
    latin_tokens = _latin_tokens(text)
    residue_hits = [
        token
        for token in latin_tokens
        if not _is_whitelisted_latin(token, cfg, source_allowlist)
        and token.lower() in {w.lower() for w in cfg.residue_words}
    ]

    if ratio < 0.55:
        product_only_latin = latin_tokens and all(
            _is_whitelisted_latin(token, cfg, source_allowlist) for token in latin_tokens
        )
        arabic_words = ARABIC_WORD_RE.findall(text)
        if product_only_latin and len(arabic_words) >= 4 and not residue_hits:
            warnings.append(f"arabic_ratio_low_but_product_heavy:{ratio:.3f}")
        else:
            errors.append(f"arabic_ratio_too_low:{ratio:.3f}")

    for token in latin_tokens:
        if _is_whitelisted_latin(token, cfg, source_allowlist):
            continue
        low = token.lower()
        if low in must_translate:
            errors.append(f"must_translate_latin:{token}")
        elif low in {w.lower() for w in cfg.residue_words}:
            errors.append(f"indonesian_residue:{token}")
        else:
            warnings.append(f"latin_unknown:{token}")

    for entity in item.entities_found:
        if entity in text or entity.lower() in text.lower():
            continue
        if _entity_satisfied_by_translation(entity, text, cfg):
            warnings.append(f"entity_translated_ok:{entity}")
            continue
        errors.append(f"missing_entity:{entity}")

    entity_conf = cfg.confusions.get("entity_confusions", {})
    source_entities = set(item.entities_found)
    for src_entity, forbidden_list in entity_conf.items():
        if src_entity not in source_entities:
            continue
        for forbidden in forbidden_list:
            if _is_entity_drift(
                src_entity,
                forbidden,
                text,
                source_entities,
                item.source_idn,
                cfg.entities,
            ):
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
            if any(_contains_forbidden_phrase(lower_tgt, f) for f in forbidden):
                errors.append(f"term_confusion:{term_name}")

    for action_name, spec in cfg.action_polarity.get("actions", {}).items():
        if action_name not in item.actions_found:
            continue
        expected = [e.lower() for e in spec.get("ar_expected", [])]
        forbidden = [f.lower() for f in spec.get("ar_forbidden", [])]
        if expected and not any(e in lower_tgt for e in expected):
            warnings.append(f"action_missing:{action_name}")
        if not forbidden:
            continue
        if action_name == "memblokir" and _dual_block_unblock_source(item) and _has_unblock_phrase(text):
            warnings.append("dual_block_unblock_allowed")
            continue
        if action_name in ("activate", "deactivate") and _dual_activate_deactivate_source(item):
            if _has_activate_phrase(text) and _has_deactivate_phrase(text):
                warnings.append("dual_activate_deactivate_allowed")
                continue
        if forbidden and any(_contains_forbidden_phrase(lower_tgt, f) for f in forbidden):
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
        "latin_tokens": latin_tokens,
    }


def apply_postprocess(item: TranslationItem, cfg: PipelineConfig) -> None:
    item.msa_clean = normalize_unicode(item.msa_raw)
    item.msa_norm_for_embedding = normalize_for_embedding(item.msa_raw)


def validate_json_payload(payload: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(payload)
    except json.JSONDecodeError as exc:
        return False, str(exc)
