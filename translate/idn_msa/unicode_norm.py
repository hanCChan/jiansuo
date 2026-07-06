from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")
MULTISPACE = re.compile(r"\s+")
TATWEEL = "\u0640"
HARAKAT = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    text = MULTISPACE.sub(" ", text)
    return text.strip()


def normalize_for_embedding(text: str) -> str:
    text = normalize_unicode(text)
    text = text.replace(TATWEEL, "")
    text = HARAKAT.sub("", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text
