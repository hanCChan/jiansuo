from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    entities: list[str]
    latin_whitelist: list[str]
    glossary: dict[str, Any]
    confusions: dict[str, Any]
    action_polarity: dict[str, Any]
    residue_words: list[str]
    config_dir: Path


def load_triage_config(config_dir: Path | None = None) -> dict[str, Any]:
    root = config_dir or Path(__file__).resolve().parent.parent / "config"
    with open(root / "triage.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: Path | None = None) -> PipelineConfig:
    root = config_dir or Path(__file__).resolve().parent.parent / "config"
    with open(root / "entities.yaml", encoding="utf-8") as f:
        entities_cfg = yaml.safe_load(f)
    with open(root / "glossary.yaml", encoding="utf-8") as f:
        glossary = yaml.safe_load(f)
    with open(root / "confusions.yaml", encoding="utf-8") as f:
        confusions = yaml.safe_load(f)
    with open(root / "action_polarity.yaml", encoding="utf-8") as f:
        action_polarity = yaml.safe_load(f)
    residue_words = [
        line.strip()
        for line in (root / "residue_id_words.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return PipelineConfig(
        entities=entities_cfg["entities"],
        latin_whitelist=entities_cfg["latin_whitelist"],
        glossary=glossary,
        confusions=confusions,
        action_polarity=action_polarity,
        residue_words=residue_words,
        config_dir=root,
    )
