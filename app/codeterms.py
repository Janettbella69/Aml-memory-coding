"""Identifier-aware terms for the coding track (v1.1, CODE_TERMS).

A camelCase or PascalCase identifier is one token to FTS5's unicode61 tokenizer and to search.word_terms, so a task that
says "user name" never meets a transcript that says `getUserName`. With CODE_TERMS the words inside such identifiers are
added on both sides: at Add time appended to the indexed text (the stored content stays verbatim; memories_fts is an
external-content index, so only the index sees the extra words), at Search time added to the query terms. snake_case,
kebab-case, dotted and slashed names already split at those characters on both sides and are left alone.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
INNER_BOUNDARY = re.compile(r"[a-z][A-Z]|[A-Za-z][0-9]|[0-9][A-Za-z]|[A-Z]{2}[a-z]")
PARTS_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
MAX_EXTRA = 2000   # extra words per turn (a tool result can dump a whole file)


def split_identifier(token: str) -> list[str]:
    """Lower-cased words inside one identifier, [] when it has no inner boundary:
    getUserName -> [get, user, name]; HTTPResponse2 -> [http, response, 2]; ndarithmetic -> []; get_user_name -> []."""
    if not INNER_BOUNDARY.search(token):
        return []
    parts = [p.lower() for p in PARTS_RE.findall(token)]
    return parts if len(parts) > 1 else []


def extra_words(text: str, limit: int = MAX_EXTRA) -> list[str]:
    """Distinct inner words of every identifier in `text` that has an inner boundary, in first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in TOKEN_RE.finditer(text):
        for word in split_identifier(m.group(0)):
            if word not in seen:
                seen.add(word)
                out.append(word)
                if len(out) >= limit:
                    return out
    return out


def index_text(content: str) -> str:
    """What the word index sees for a turn: the content, then its identifiers' inner words on a line of their own."""
    words = extra_words(content)
    return content + "\n" + " ".join(words) if words else content
