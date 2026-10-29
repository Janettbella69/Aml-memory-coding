"""Local contract smoke: python scripts/smoke_test.py http://127.0.0.1:8000 [api_key]"""
from __future__ import annotations

import json
import sys
import urllib.request


def call(base: str, path: str, payload: dict | None, key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Api-Key"] = key
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(payload).encode() if payload else None,
                                 headers=headers, method="POST" if payload else "GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    key = sys.argv[2] if len(sys.argv) > 2 else None
    assert call(base, "/health", None, None)["status"] == "ok"
    added = call(base, "/v1/memories/add", {
        "request_id": "smoke:chunk-0", "user_id": "smoke:user", "session_id": "smoke:session",
        "messages": [{"role": "user", "timestamp": 1704067200000, "content": "I adopted a cat named Luna."}],
    }, key)
    assert added["success"] is True and added["request_id"] == "smoke:chunk-0"
    found = call(base, "/v1/memories/search", {"query": "What is my cat called?", "user_id": "smoke:user", "top_k": 5}, key)
    assert found["data"] and "Luna" in found["data"][0]["content"], found
    print("smoke passed:", found["data"][0]["content"])


if __name__ == "__main__":
    main()
