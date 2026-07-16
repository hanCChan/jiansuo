"""Shared helpers for dialogue XXX semantic fill pipeline."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

from idn_msa.dialogue_mask import SENSITIVE_TOKEN_RE

# --- hint type constants ---
HINT_RATE = "rate"
HINT_TENOR = "tenor"
HINT_AMOUNT = "amount"
HINT_DATE = "date"
HINT_COUNT = "count"
HINT_PII_CARD = "pii_card"
HINT_PII_PHONE = "pii_phone"
HINT_PII_NAME = "pii_name"
HINT_PII_GENERIC = "pii_generic"
HINT_UNKNOWN = "unknown"

HINT_TYPES = (
    HINT_RATE,
    HINT_TENOR,
    HINT_AMOUNT,
    HINT_DATE,
    HINT_COUNT,
    HINT_PII_CARD,
    HINT_PII_PHONE,
    HINT_PII_NAME,
    HINT_PII_GENERIC,
    HINT_UNKNOWN,
)

PII_HINTS = {HINT_PII_CARD, HINT_PII_PHONE, HINT_PII_NAME, HINT_PII_GENERIC}
NUMERIC_PII_HINTS = {HINT_PII_CARD, HINT_PII_PHONE, HINT_PII_GENERIC}

RULE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (HINT_RATE, re.compile(
        r"koma|persen|percent|فاصلة|في المئة|نسبة|interest|bunga|faedah",
        re.I,
    )),
    (HINT_TENOR, re.compile(
        r"bulan|tenor|شهر|أشهر|شهور|month|angsuran|cicilan|installment",
        re.I,
    )),
    (HINT_AMOUNT, re.compile(
        r"ribu|juta|rupiah|روبية|ألف|مليون|amount|biaya|angsuran|nominal",
        re.I,
    )),
    (HINT_DATE, re.compile(
        r"maret|april|mei|juni|juli|agustus|september|oktober|november|desember|"
        r"januari|februari|tanggal|tgl|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|"
        r"سبتمبر|أكتوبر|نوفمبر|ديسمبر|يناير|فبراير|تاريخ",
        re.I,
    )),
    (HINT_COUNT, re.compile(
        r"\bkali\b|مرات|محاولة|hari kerja|working day|menit|دقيقة",
        re.I,
    )),
    (HINT_PII_CARD, re.compile(
        r"nomor kartu|kartu kredit|kartu debit|رقم البطاقة|بطاقة",
        re.I,
    )),
    (HINT_PII_PHONE, re.compile(
        r"handphone|nomor hp|nomor telepon|هاتف|جوال|hp\b|手机号|电话",
        re.I,
    )),
    (HINT_PII_NAME, re.compile(
        r"\bnama\b|siapa nama|namanya|姓名|名字|اسم",
        re.I,
    )),
    (
        HINT_PII_GENERIC,
        re.compile(
            r"nomor rekening|rekening|akun|account|رقم الحساب|حساب|账号",
            re.I,
        ),
    ),
]

SYNTHETIC_ARABIC_NAMES = (
    "أحمد",
    "فاطمة",
    "علي",
    "سارة",
    "حسن",
    "نورا",
    "يوسف",
    "مريم",
    "خالد",
    "ليلى",
)

LONG_DIGIT_RE = re.compile(r"\d{10,}")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _synthetic_seed(*parts: str) -> int:
    blob = "|".join(parts)
    return int(hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8], 16)


def xxx_mask_length(token: str) -> int:
    """Approximate digit length from XXX-style mask token."""
    x_count = len(re.findall(r"X", token, flags=re.I))
    if x_count >= 6:
        return 10 if x_count <= 8 else 12
    if x_count >= 4:
        return 6
    if x_count >= 3:
        return 3
    return 2


def generate_synthetic_digits(length: int, seed: int) -> str:
    rng = random.Random(seed)
    length = max(2, min(length, 16))
    digits = [str(rng.randint(1, 9))]
    digits.extend(str(rng.randint(0, 9)) for _ in range(length - 1))
    return "".join(digits)


def generate_synthetic_phone(seed: int) -> str:
    rng = random.Random(seed)
    prefix = rng.choice(["0812", "0813", "0856", "0878"])
    tail = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return prefix + tail


def generate_synthetic_card(seed: int) -> str:
    rng = random.Random(seed)
    first = str(rng.randint(4, 6))
    rest = "".join(str(rng.randint(0, 9)) for _ in range(15))
    return first + rest


def generate_synthetic_name(seed: int) -> str:
    return SYNTHETIC_ARABIC_NAMES[seed % len(SYNTHETIC_ARABIC_NAMES)]


def suggest_synthetic_value(
    token: str,
    hint_type: str,
    *,
    cluster_id: str,
    turn: int,
    token_idx: int,
) -> dict[str, Any]:
    """Deterministic synthetic suggestion for Kimi (training-safe fake values)."""
    seed = _synthetic_seed(cluster_id, str(turn), str(token_idx), token, hint_type)
    if hint_type == HINT_PII_PHONE:
        value = generate_synthetic_phone(seed)
        return {
            "policy": "random_phone",
            "suggested_value": value,
            "digit_length": len(value),
        }
    if hint_type == HINT_PII_CARD:
        value = generate_synthetic_card(seed)
        return {
            "policy": "random_card",
            "suggested_value": value,
            "digit_length": len(value),
        }
    if hint_type == HINT_PII_NAME:
        value = generate_synthetic_name(seed)
        return {
            "policy": "random_name",
            "suggested_value": value,
        }
    if hint_type == HINT_PII_GENERIC:
        length = xxx_mask_length(token)
        if length >= 6:
            value = generate_synthetic_digits(length, seed)
            return {
                "policy": "random_account_digits",
                "suggested_value": value,
                "digit_length": length,
            }
    length = xxx_mask_length(token)
    value = generate_synthetic_digits(length, seed)
    return {
        "policy": "random_digits",
        "suggested_value": value,
        "digit_length": length,
    }


def find_xxx_tokens(text: str) -> list[str]:
    return SENSITIVE_TOKEN_RE.findall(text or "")


def has_xxx(text: str) -> bool:
    return bool(find_xxx_tokens(text))


def strip_xxx(text: str) -> str:
    return SENSITIVE_TOKEN_RE.sub("", text or "")


def classify_hint_type(*texts: str) -> str:
    """Classify XXX slot from surrounding IDN / zh / MSA context."""
    blob = " ".join(t for t in texts if t).strip()
    if not blob:
        return HINT_UNKNOWN

    hits: list[str] = []
    for hint_type, pattern in RULE_PATTERNS:
        if pattern.search(blob):
            hits.append(hint_type)

    if hits:
        for preferred in (
            HINT_RATE,
            HINT_TENOR,
            HINT_AMOUNT,
            HINT_DATE,
            HINT_COUNT,
            HINT_PII_CARD,
            HINT_PII_PHONE,
            HINT_PII_NAME,
        ):
            if preferred in hits:
                return preferred
        return hits[0]

    xxx_count = len(find_xxx_tokens(blob))
    if xxx_count >= 3:
        return HINT_PII_GENERIC
    return HINT_UNKNOWN


@dataclass
class TurnRef:
    dialogue_id: int
    turn: int
    role: str
    content: str
    content_zh: str
    content_msa: str
    row_id: int | None = None

    def hint_type(self) -> str:
        return classify_hint_type(self.content, self.content_zh, self.content_msa)

    def to_dict(self, *, include_context_fields: bool = True) -> dict[str, Any]:
        row: dict[str, Any] = {
            "turn": self.turn,
            "role": self.role,
            "content": self.content,
            "content_zh": self.content_zh,
            "content_msa": self.content_msa,
            "xxx_tokens_in_order": find_xxx_tokens(self.content_msa),
            "hint_type": self.hint_type(),
        }
        if self.row_id is not None:
            row["row_id"] = self.row_id
        return row


@dataclass
class XxxCluster:
    dialogue_id: int
    cluster_idx: int
    turns: list[TurnRef]
    gap_max: int = 3

    @property
    def cluster_id(self) -> str:
        return f"{self.dialogue_id}_c{self.cluster_idx}"

    def topic_hint(self) -> str:
        types = [t.hint_type() for t in self.turns]
        dominant = max(set(types), key=types.count)
        return dominant

    def slot_hints(self) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        token_idx = 0
        for turn in self.turns:
            hint_type = turn.hint_type()
            for token in find_xxx_tokens(turn.content_msa):
                token_idx += 1
                synthetic = suggest_synthetic_value(
                    token,
                    hint_type,
                    cluster_id=self.cluster_id,
                    turn=turn.turn,
                    token_idx=token_idx,
                )
                hints.append(
                    {
                        "turn": turn.turn,
                        "token_index": token_idx,
                        "token": token,
                        "hint_type": hint_type,
                        **synthetic,
                    }
                )
        return hints

    def to_kimi_payload(self, context_window: int = 1) -> dict[str, Any]:
        turn_nums = {t.turn for t in self.turns}
        all_turns = sorted(self.turns, key=lambda t: t.turn)
        min_turn = all_turns[0].turn
        max_turn = all_turns[-1].turn

        # local context is injected by caller using full dialogue turns map
        return {
            "cluster_id": self.cluster_id,
            "dialogue_id": self.dialogue_id,
            "topic_hint": self.topic_hint(),
            "slot_hints": self.slot_hints(),
            "turns_to_fill": [t.to_dict() for t in self.turns],
            "context_window": context_window,
            "turn_range": [min_turn, max_turn],
            "consistency_rules": [
                "Only keep consistency within this local cluster.",
                "Do not force the same synthetic value across unrelated clusters.",
                "For pii_card/pii_phone/account: use random synthetic digits that look realistic.",
                "Different XXX slots may use different random values unless context clearly refers to the same item.",
            ],
        }


def load_turn_refs(dialogues: list[dict[str, Any]]) -> dict[int, list[TurnRef]]:
    by_dialogue: dict[int, list[TurnRef]] = {}
    for dlg in dialogues:
        did = int(dlg["dialogue_id"])
        refs: list[TurnRef] = []
        for turn in dlg.get("turns", []):
            refs.append(
                TurnRef(
                    dialogue_id=did,
                    turn=int(turn.get("turn", 0)),
                    role=str(turn.get("role") or ""),
                    content=str(turn.get("content") or "").strip(),
                    content_zh=str(turn.get("content_zh") or "").strip(),
                    content_msa=str(turn.get("content_msa") or "").strip(),
                    row_id=turn.get("row_id"),
                )
            )
        by_dialogue[did] = sorted(refs, key=lambda r: r.turn)
    return by_dialogue


def build_local_context(
    dialogue_turns: list[TurnRef],
    cluster: XxxCluster,
    window: int = 1,
) -> list[dict[str, Any]]:
    cluster_turns = {t.turn for t in cluster.turns}
    min_t = min(cluster_turns)
    max_t = max(cluster_turns)
    ctx: list[dict[str, Any]] = []
    for ref in dialogue_turns:
        if ref.turn in cluster_turns:
            continue
        if min_t - window <= ref.turn < min_t or max_t < ref.turn <= max_t + window:
            ctx.append(
                {
                    "turn": ref.turn,
                    "role": ref.role,
                    "content_msa": ref.content_msa,
                }
            )
    return ctx


def cluster_xxx_turns(
    dialogue_turns: list[TurnRef],
    *,
    gap_max: int = 3,
) -> list[XxxCluster]:
    """Group nearby XXX turns into local clusters."""
    xxx_turns = [t for t in dialogue_turns if has_xxx(t.content_msa)]
    if not xxx_turns:
        return []

    clusters: list[list[TurnRef]] = []
    current: list[TurnRef] = [xxx_turns[0]]

    for turn in xxx_turns[1:]:
        prev = current[-1]
        if turn.turn - prev.turn <= gap_max:
            current.append(turn)
        else:
            clusters.append(current)
            current = [turn]
    clusters.append(current)

    dialogue_id = dialogue_turns[0].dialogue_id
    return [
        XxxCluster(dialogue_id=dialogue_id, cluster_idx=i, turns=chunk, gap_max=gap_max)
        for i, chunk in enumerate(clusters)
    ]


def build_all_clusters(
    dialogues: list[dict[str, Any]],
    *,
    gap_max: int = 3,
) -> tuple[dict[int, list[TurnRef]], list[XxxCluster]]:
    by_dialogue = load_turn_refs(dialogues)
    all_clusters: list[XxxCluster] = []
    for did in sorted(by_dialogue):
        clusters = cluster_xxx_turns(by_dialogue[did], gap_max=gap_max)
        all_clusters.extend(clusters)
    return by_dialogue, all_clusters
