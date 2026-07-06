from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TranslationCache:
    """Global IDN -> MSA cache keyed by exact source Indonesian text."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        if path and path.exists():
            self.load(path)

    def load(self, path: Path | None = None) -> None:
        path = path or self.path
        if not path or not path.exists():
            return
        self._data.clear()
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    src = row.get("source_idn") or row.get("idn")
                    msa = row.get("msa_raw") or row.get("msa")
                    if src and msa:
                        self._data[src] = msa
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._data.update(payload)

    def save(self, path: Path | None = None) -> None:
        path = path or self.path
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for src, msa in sorted(self._data.items()):
                f.write(json.dumps({"source_idn": src, "msa": msa}, ensure_ascii=False) + "\n")

    def get(self, source_idn: str) -> str | None:
        return self._data.get(source_idn)

    def set(self, source_idn: str, msa: str) -> None:
        if source_idn and msa:
            self._data[source_idn] = msa

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._data)}
