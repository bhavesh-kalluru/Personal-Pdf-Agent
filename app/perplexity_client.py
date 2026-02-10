from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class PerplexityClient:
    """Minimal Perplexity client (OpenAI-compatible chat completions endpoint)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "")
        self.model = model or os.getenv("PERPLEXITY_MODEL", "sonar-pro")
        self.base_url = "https://api.perplexity.ai/chat/completions"

    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    def web_answer(self, user_question: str) -> str:
        if not self.is_configured():
            raise RuntimeError("Perplexity is not configured. Set PERPLEXITY_API_KEY in .env.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You answer using web search. Be concise and include sources if available."},
                {"role": "user", "content": user_question},
            ],
            "temperature": 0.2,
        }

        r = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"Perplexity API error ({r.status_code}): {r.text}")

        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return str(data)
