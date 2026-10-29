"""One bounded chat-completion call to an OpenAI-compatible endpoint, shared by the Add-time session summariser
(app/summarize.py) and the Search-time card selector (app/cardselect.py). The open-source rules allow `gpt-4o-mini`
only; via OpenRouter the model id is `openai/gpt-4o-mini` (the same model, routed). Any failure returns None and the
caller keeps its deterministic behaviour. No request or response text is logged."""
from __future__ import annotations

import http.client
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

log = logging.getLogger("aml_memory.llm")
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def credentials() -> tuple[str, str]:
    """(api key, base url) from the environment: OPENAI_API_KEY, OPENAI_BASE_URL."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    return key, base


def chat(model: str, api_key: str, base_url: str, system: str, user: str, timeout: float = 30.0,
         max_tokens: int = 400, json_object: bool = False) -> Optional[str]:
    body: dict = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_object:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.load(resp)
        text = out["choices"][0]["message"]["content"]
        return text if isinstance(text, str) else None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, KeyError, IndexError, TypeError,
            ValueError, json.JSONDecodeError) as exc:
        log.warning("chat call failed (%s)", type(exc).__name__)   # error type only, never text
        return None
