"""Real LLM clients. NO mocks in any evaluated path.

Division of labor (per design): the CHEAP local model (Ollama) does the bulk summarization;
a FRONTIER model (Anthropic Claude, preferred) is used only for controls/upper-bounds (H5) or
as the client in end-to-end eval. Both implement the `LLMClient` protocol (generate).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


def load_env(env_path: str | Path | None = None) -> None:
    """Minimal .env loader: populate os.environ from KEY=VALUE lines (no overwrite)."""
    p = Path(env_path) if env_path else Path(__file__).resolve().parents[2] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v


class OllamaClient:
    """Cheap, local, FREE model (default summarizer). Real model — not a mock."""

    def __init__(self, model: str = "qwen2.5-coder:3b", host: str = "http://localhost:11434",
                 timeout: int = 180):
        self.model = model
        self.host = host
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        # 0.2 was this client's historical default; pass temperature=0.0 explicitly for
        # exact-T0 protocols (the originally shipped exp-015 run used the 0.2 default).
        body = {
            "model": self.model, "prompt": prompt, "stream": False,
            "think": False,  # gpt-oss & reasoning models: emit the answer, not reasoning
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        return (r.get("response") or "").strip()


class AnthropicClient:
    """Frontier model (preferred per project note). Real API; key from env/.env."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        load_env()
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key or key.startswith("sk-ant-your"):
            raise RuntimeError("ANTHROPIC_API_KEY not set (real key required; no mock)")
        import anthropic
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.last_model_id: str | None = None      # dated snapshot the API actually resolved to
        self.last_stop_reason: str | None = None    # "end_turn" vs "max_tokens" (truncation detection)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float | None = None) -> str:
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if temperature is not None:
            kwargs["temperature"] = temperature
        msg = self.client.messages.create(**kwargs)
        self.last_model_id = getattr(msg, "model", None)
        self.last_stop_reason = getattr(msg, "stop_reason", None)
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()

    def count_tokens(self, prompt: str) -> int | None:
        """Real Anthropic input-token count (not a cl100k proxy). None if unavailable."""
        try:
            r = self.client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": prompt}])
            return getattr(r, "input_tokens", None)
        except Exception:
            return None


def cheap_client(model: str = "qwen2.5-coder:3b") -> OllamaClient:
    return OllamaClient(model=model)


def frontier_client(model: str = "claude-sonnet-4-6") -> AnthropicClient:
    """Anthropic Claude — preferred frontier model for controls/clients."""
    return AnthropicClient(model=model)
