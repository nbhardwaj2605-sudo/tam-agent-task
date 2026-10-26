"""Thin LLM client. Pick a provider with environment variables (see .env.example).

Supported out of the box:
  LLM_PROVIDER=ollama     free, local (https://ollama.com) - no API key needed
  LLM_PROVIDER=anthropic  needs ANTHROPIC_API_KEY and `pip install anthropic`
  LLM_PROVIDER=openai     needs OPENAI_API_KEY and `pip install openai`

You may replace this file, add native tool calling, or use a framework
(LangGraph, LlamaIndex, etc.). Keep the provider configurable via env vars.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

# minimal .env loader (no extra dependency)
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.split("#", 1)[0].strip()
        if "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("LLM_MODEL", {
            "ollama": "llama3.1:8b", "anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}[self.provider])
        self.calls = 0  # simple counter you can report for cost/latency
        self.input_tokens = 0
        self.output_tokens = 0
        self._client = None  # SDK client is created once and reused

    def chat(self, messages: list[dict], system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, json_mode: bool = False) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}] -> assistant text.

        json_mode asks the provider for a JSON object where supported (Ollama `format`,
        OpenAI `response_format`). The agent still parses defensively.
        Rate-limit / overload errors (common on free tiers) are retried with backoff.
        LLM_MAX_TOKENS raises the output cap (needed for "thinking" models such as Gemini 2.5,
        whose hidden reasoning counts against max_tokens).
        """
        max_tokens = max(max_tokens, int(os.getenv("LLM_MAX_TOKENS", "0") or 0))
        retries = int(os.getenv("LLM_RETRIES", "6"))
        for attempt in range(retries + 1):
            try:
                return self._chat_once(messages, system, max_tokens, temperature, json_mode)
            except Exception as e:  # noqa: BLE001 - provider SDKs raise different types
                msg = repr(e).lower()
                transient = any(k in msg for k in ("429", "rate", "quota", "overloaded", "503", "529",
                                                   "timeout", "temporarily", "resource_exhausted"))
                if not transient or attempt == retries:
                    raise
                wait = min(60, 5 * 2 ** attempt)
                print(f"  [llm] transient error ({type(e).__name__}); retrying in {wait}s")
                time.sleep(wait)

    def _chat_once(self, messages, system, max_tokens, temperature, json_mode) -> str:
        self.calls += 1
        if self.provider == "anthropic":
            import anthropic
            self._client = self._client or anthropic.Anthropic()
            resp = self._client.messages.create(model=self.model, max_tokens=max_tokens, temperature=temperature,
                                                system=system or "", messages=messages)
            self.input_tokens += resp.usage.input_tokens
            self.output_tokens += resp.usage.output_tokens
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if self.provider == "openai":
            from openai import OpenAI
            self._client = self._client or OpenAI()
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            kw = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = self._client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens,
                                                        temperature=temperature, **kw)
            if resp.usage:
                self.input_tokens += resp.usage.prompt_tokens
                self.output_tokens += resp.usage.completion_tokens
            return resp.choices[0].message.content or ""
        if self.provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            msgs = ([{"role": "system", "content": system}] if system else []) + messages
            # Ollama's default context window (2-4k) is too small for a multi-step agent conversation.
            opts = {"temperature": temperature, "num_predict": max_tokens,
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384"))}
            payload = {"model": self.model, "messages": msgs, "stream": False, "options": opts}
            if json_mode:
                payload["format"] = "json"
            req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
            self.input_tokens += data.get("prompt_eval_count", 0)
            self.output_tokens += data.get("eval_count", 0)
            return data["message"]["content"]
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
