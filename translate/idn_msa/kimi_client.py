from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class KimiClient:
    def __init__(
        self,
        base_url: str = "http://10.16.137.2:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Kimi-K2.6-CT-FP8KV",
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.max_retries = max_retries

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Kimi API attempt %s failed: %s", attempt, exc)
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Kimi API failed after retries: {last_error}")

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Kimi API attempt %s failed: %s", attempt, exc)
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Kimi API failed after retries: {last_error}")
