from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _require_text(name: str, value: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{name} must contain at least one non-whitespace character")
    return value


def parse_timestamp(value: Any) -> Optional[int]:
    """Unix milliseconds from whatever the adapter sent. The contract says "optional Unix milliseconds"; a null, a
    placeholder such as "Unknown" (ScriptMem's `session_N_date_time` default), a float, a numeric string or an ISO 8601
    string must never turn a whole Add into a 422 (permanent 4xx responses are not retried by the platform)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        ms = int(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            ms = int(float(text))
        except ValueError:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
    if abs(ms) >= 10 ** 17:      # nanoseconds
        ms //= 10 ** 6
    elif abs(ms) >= 10 ** 14:    # microseconds
        ms //= 10 ** 3
    return ms


def _parts_text(value: Any) -> str:
    """Text of a string or of OpenAI-style content parts (text parts joined, image parts dropped)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        pieces = []
        for part in value:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text" and part.get("text"):
                pieces.append(str(part.get("text")))
        return "\n".join(pieces)
    return str(value)


class MemoryMessage(BaseModel):
    """One turn of the source conversation. `content` may be a string or a list of content parts;
    only text parts are kept (textual track), image parts are dropped. Parsing is liberal: an odd role, a null
    content or an unparseable timestamp degrades that field, it never rejects the request."""
    model_config = ConfigDict(extra="ignore")
    role: str = "user"
    content: Union[str, List[Any]] = ""
    timestamp: Optional[int] = None  # unix milliseconds, optional per the contract

    @field_validator("role", mode="before")
    @classmethod
    def _role(cls, v: Any) -> str:
        text = "" if v is None else str(v).strip()
        return text or "user"

    @field_validator("content", mode="before")
    @classmethod
    def _content(cls, v: Any) -> Any:
        if v is None:
            return ""
        if isinstance(v, (str, list)):
            return v
        if isinstance(v, dict):
            return [v]
        return str(v)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _timestamp(cls, v: Any) -> Optional[int]:
        return parse_timestamp(v)

    def text(self) -> str:
        return _parts_text(self.content)


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    request_id: str
    messages: List[MemoryMessage] = Field(min_length=1)
    user_id: str
    session_id: str

    @field_validator("request_id")
    @classmethod
    def _rid(cls, v: str) -> str:
        return _require_text("request_id", v)

    @field_validator("user_id")
    @classmethod
    def _uid(cls, v: str) -> str:
        return _require_text("user_id", v)

    @field_validator("session_id")
    @classmethod
    def _sid(cls, v: str) -> str:
        return _require_text("session_id", v)


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    """Liberal on everything but `user_id`: a blank query, options sent as objects or one string, or an out-of-range
    top_k still gets a 200 with evidence (a Search 4xx is permanent for the platform and costs the question)."""
    model_config = ConfigDict(extra="ignore")
    query: str = ""
    options: Optional[List[str]] = None
    user_id: str
    top_k: int = 100

    @field_validator("query", mode="before")
    @classmethod
    def _q(cls, v: Any) -> str:
        return _parts_text(v)

    @field_validator("options", mode="before")
    @classmethod
    def _options(cls, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        elif isinstance(v, dict):
            v = [f"{k}. {val}" for k, val in v.items()]
        if not isinstance(v, (list, tuple)):
            return None
        out: List[str] = []
        for item in v:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("key") or item.get("id") or "").strip()
                body = _parts_text(item.get("text") or item.get("content") or item.get("value") or item.get("option"))
                item = f"{label}. {body}" if label and body else (body or label)
            text = "" if item is None else str(item).strip()
            if text:
                out.append(text)
        return out or None

    @field_validator("top_k", mode="before")
    @classmethod
    def _top_k(cls, v: Any) -> int:
        try:
            k = int(float(v))
        except (TypeError, ValueError):
            return 100
        return max(1, min(1000, k))

    @field_validator("user_id")
    @classmethod
    def _uid(cls, v: str) -> str:
        return _require_text("user_id", v)


class MemoryResult(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    created_at: Optional[str] = None


class SearchResponse(BaseModel):
    data: List[MemoryResult]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
