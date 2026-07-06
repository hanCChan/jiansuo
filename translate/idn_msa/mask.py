from __future__ import annotations

import re
from typing import Iterable

from .config_loader import PipelineConfig


def build_entity_maps(entities: Iterable[str]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    sorted_entities = sorted(set(entities), key=len, reverse=True)
    forward: list[tuple[str, str]] = []
    reverse: dict[str, str] = {}
    for idx, entity in enumerate(sorted_entities):
        placeholder = f"<ENT_{idx:02d}>"
        forward.append((entity, placeholder))
        reverse[placeholder] = entity
    return forward, reverse


def mask_text(text: str, cfg: PipelineConfig) -> tuple[str, list[str]]:
    forward, _ = build_entity_maps(cfg.entities)
    found: list[str] = []
    masked = text
    for entity, placeholder in forward:
        pattern = re.compile(re.escape(entity), flags=re.IGNORECASE)
        if pattern.search(masked):
            found.append(entity)
            masked = pattern.sub(placeholder, masked)
    return masked, found


def restore_text(text: str, cfg: PipelineConfig) -> str:
    _, reverse = build_entity_maps(cfg.entities)
    restored = text
    for placeholder, entity in reverse.items():
        restored = restored.replace(placeholder, entity)
    return restored


def extract_terms(text: str, cfg: PipelineConfig) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for term_name, spec in cfg.glossary.get("terms", {}).items():
        for trigger in spec.get("id_triggers", []):
            if trigger.lower() in lower:
                found.append(term_name)
                break
    return found


def extract_actions(text: str, cfg: PipelineConfig) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for action_name, spec in cfg.action_polarity.get("actions", {}).items():
        for pattern in spec.get("id_patterns", []):
            if pattern.lower() in lower:
                found.append(action_name)
                break
    return found


def annotate_item(text: str, cfg: PipelineConfig) -> tuple[str, list[str], list[str], list[str]]:
    masked, entities = mask_text(text, cfg)
    terms = extract_terms(text, cfg)
    actions = extract_actions(text, cfg)
    return masked, entities, terms, actions
