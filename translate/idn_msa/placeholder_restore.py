from __future__ import annotations

import re

from .config_loader import PipelineConfig
from .mask import _entity_pattern, restore_text

ENT_RE = re.compile(r"<ENT[^>]*>", re.IGNORECASE)
GENERIC_TAG_RE = re.compile(
    r"^(ORG|SRV|ENT|VAL|BANK|POCKET|FLZZ|NUM|IDX)_?\d*$",
    re.IGNORECASE,
)
LATIN_TOKEN_RE = re.compile(
    r"[A-Z]{2,}(?:[-/][A-Z0-9]+)*"
    r"|[A-Z][a-z]+[A-Z][A-Za-z0-9]*"
    r"|[A-Za-z]+[-/][A-Za-z0-9]+"
    r"|[A-Za-z]*\d+[A-Za-z]*"
    r"|\be[A-Z][A-Za-z]+\b"
    r"|\be-[A-Za-z]+\b",
)
EXTRA_STOP_LATIN = {
    "berapa",
    "di",
    "mana",
    "apa",
    "kapan",
    "siapa",
    "mengapa",
    "bisakah",
    "apakah",
    "nama",
    "dan",
    "atau",
    "untuk",
    "dari",
    "pada",
    "yang",
    "jika",
    "saat",
    "sudah",
    "masih",
    "dengan",
    "melalui",
    "top",
    "up",
    "online",
    "status",
    "gratis",
    "kena",
    "unprinted",
    "refund",
    "link",
    "produk",
    "tabungan",
    "tunai",
    "tarik",
    "cek",
    "saldo",
    "biaya",
    "proses",
    "aturan",
    "keluhan",
    "cetak",
    "mutasi",
    "transaksi",
    "rekening",
    "relawan",
    "gagal",
    "kembali",
    "lama",
    "aktif",
    "tutup",
    "buka",
    "data",
    "saat",
    "mulai",
    "berlaku",
    "mobile",
    "form",
    "login",
    "check",
    "teller",
    "layanan",
    "cabang",
    "reservasi",
    "limit",
    "naik",
    "proses",
}


def _residue_words(cfg: PipelineConfig) -> set[str]:
    return {w.lower() for w in cfg.residue_words} | EXTRA_STOP_LATIN


def _is_residue_token(token: str, residue: set[str]) -> bool:
    return token.lower() in residue


def _is_product_like(token: str, residue: set[str]) -> bool:
    if _is_residue_token(token, residue):
        return False
    if token.isupper() and len(token) >= 2:
        return True
    if re.search(r"[a-z][A-Z]|[0-9]", token):
        return True
    if token[0].isupper() and len(token) >= 4:
        return True
    return False


def _ordered_source_latin(source: str, cfg: PipelineConfig) -> list[str]:
    residue = _residue_words(cfg)
    spans: list[tuple[int, int, str]] = []

    for entity in sorted(cfg.entities, key=len, reverse=True):
        pattern = _entity_pattern(entity)
        for match in pattern.finditer(source):
            spans.append((match.start(), match.end(), match.group()))

    for match in LATIN_TOKEN_RE.finditer(source):
        text = match.group()
        if _is_product_like(text, residue):
            spans.append((match.start(), match.end(), text))

    for match in re.finditer(r"\b[A-Z][a-z]{3,}\b", source):
        text = match.group()
        if _is_product_like(text, residue):
            spans.append((match.start(), match.end(), text))

    for match in re.finditer(r"\bstatus\s+([A-Z0-9])\b", source, re.IGNORECASE):
        spans.append((match.start(1), match.end(1), match.group(1).upper()))

    for match in re.finditer(r"\blimit\s+([A-Z0-9])\b", source, re.IGNORECASE):
        spans.append((match.start(1), match.end(1), match.group(1).upper()))

    for match in re.finditer(r"\be[A-Z][A-Za-z]+\b", source):
        text = match.group()
        if not _is_residue_token(text, residue):
            spans.append((match.start(), match.end(), text))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    picked: list[str] = []
    occupied: list[tuple[int, int]] = []
    for start, end, text in spans:
        if any(not (end <= s or start >= e) for s, e in occupied):
            continue
        occupied.append((start, end))
        picked.append(text)
    return picked


def _numeric_tag_index(tag: str) -> int:
    digits = re.sub(r"\D", "", tag)
    if not digits:
        return 0
    return max(int(digits) - 1, 0)


def _resolve_tag(tag: str, source: str, cfg: PipelineConfig) -> str | None:
    if tag.upper().startswith("ORG") or tag.upper().startswith("SRV"):
        tokens = _ordered_source_latin(source, cfg)
        if tokens:
            return tokens[0]

    if GENERIC_TAG_RE.match(tag):
        return None

    candidates = [tag, tag.replace("_", " "), tag.replace("_", "-")]
    for candidate in candidates:
        if len(candidate) < 2:
            continue
        pattern = re.compile(
            re.escape(candidate).replace(r"\-", r"[-\s_]?").replace(r"\ ", r"[\s_-]?"),
            re.IGNORECASE,
        )
        match = pattern.search(source)
        if match:
            return match.group()

    if re.fullmatch(r"[A-Z0-9-]{2,}", tag):
        pattern = re.compile(rf"\b{re.escape(tag)}\b", re.IGNORECASE)
        match = pattern.search(source)
        if match:
            return match.group()

    for entity in sorted(cfg.entities, key=len, reverse=True):
        key = re.sub(r"[^A-Za-z0-9]", "", entity).lower()
        tag_key = re.sub(r"[^A-Za-z0-9]", "", tag).lower()
        if key and tag_key and (key == tag_key or key.startswith(tag_key) or tag_key.startswith(key)):
            pattern = _entity_pattern(entity)
            match = pattern.search(source)
            if match:
                return match.group()
    return None


def restore_kimi_placeholders(source_idn: str, msa: str, cfg: PipelineConfig) -> tuple[str, list[str]]:
    """Replace Kimi hallucinated <ENT_*> with real names from source."""
    notes: list[str] = []
    text = restore_text(msa, cfg)

    while ENT_RE.search(text):
        placeholders = ENT_RE.findall(text)
        changed = False

        for placeholder in placeholders:
            tag = placeholder[5:-1]
            replacement = _resolve_tag(tag, source_idn, cfg)
            if replacement:
                text = text.replace(placeholder, replacement, 1)
                notes.append(f"{placeholder}->{replacement}")
                changed = True

        if changed:
            continue

        numeric_left = [ph for ph in ENT_RE.findall(text) if re.fullmatch(r"ENT_\d+", ph[5:-1], re.I)]
        if numeric_left:
            latin_seq = _ordered_source_latin(source_idn, cfg)
            for placeholder in numeric_left:
                tag = placeholder[5:-1]
                idx = _numeric_tag_index(tag)
                replacement = None
                if idx < len(latin_seq):
                    replacement = latin_seq[idx]
                elif latin_seq:
                    replacement = latin_seq[0]
                if replacement:
                    text = text.replace(placeholder, replacement, 1)
                    notes.append(f"{placeholder}->{replacement}")
                    changed = True

        if changed:
            continue

        remaining = ENT_RE.findall(text)
        latin_seq = _ordered_source_latin(source_idn, cfg)
        for i, placeholder in enumerate(remaining):
            if i < len(latin_seq):
                replacement = latin_seq[i]
                text = text.replace(placeholder, replacement, 1)
                notes.append(f"{placeholder}->{replacement}")
                changed = True

        if not changed:
            break

    return text, notes
