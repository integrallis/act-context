"""Independent (different-family) judge for exp-015: OpenAI chat completions.

Reads OPENAI_API_KEY via dhcm_ng.llm.load_env (.env). Mirrors the dhcm_ng client interface
(generate(prompt, max_tokens, temperature)) so judges are interchangeable in regrade_run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dhcm_ng.llm import load_env  # noqa: E402


class OpenAIJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        load_env()
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 16, temperature: float = 0.0) -> str:
        kwargs = dict(model=self.model,
                      messages=[{"role": "user", "content": prompt}],
                      temperature=temperature)
        try:
            r = self.client.chat.completions.create(max_tokens=max_tokens, **kwargs)
        except Exception as e:
            if "max_tokens" not in str(e):
                raise
            # newer OpenAI models take max_completion_tokens instead
            r = self.client.chat.completions.create(max_completion_tokens=max_tokens, **kwargs)
        return r.choices[0].message.content or ""
