from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .models import AddRequest
from .embed import Embedder, from_blob, to_blob
from .timex import notes_suffix

SCHEMA_VERSION = 6  # bump whenever the table layout or FTS tokenizer changes; old files are refused, not migrated

# Statements that read like standing instructions, constraints or preferences. Tagged at Add time
# (user role only) so Search can always offer them: "rule execution" is where every system on the
# cycle-1 board collapsed. Verbs are restricted to assistant-facing behaviour so that chatter such as
# "he always does" or "don't be smug" is not tagged.
_VERBS = (r"(?:suggest|recommend|bring\ up|mention|assume|use|call|refer|apply|deploy|order|put|include|propose|"
          r"write|add|answer|reply|respond|format|give|keep|flag|check|ask|send|show|say|talk|treat|address|serve|cook|offer|run|install)")
RULE_RE = re.compile(
    r"""(?x)\b(?:
        from\ now\ on | going\ forward | in\ (?:the\ )?future |
        whenever | every\ time | each\ time | when(?:ever)?\ (?:i|you)\ (?:ask|review|write|recommend|suggest|talk|give|send|plan|cook|book) |
        please\ (?:always|never|don'?t|do\ not|stop|only|keep|avoid|call|refer|use|check|make\ sure|remember) |
        (?:don'?t|do\ not|never|always|only)\ (?:ever\ )?""" + _VERBS + r""" |
        remember\ (?:that|to|this|:) | i\ want\ you\ to | i\ need\ you\ to | i\ expect\ you\ to |
        call\ me\b | refer\ to\ me | address\ me |
        (?:i|we)\ (?:prefer|'d\ prefer|would\ prefer|like\ it\ when) |
        my\ (?:preference|rule)s? | (?:ground|house|new|one|another|first|second|third|format(?:ting)?)\ rules?\b | one\ more\ rule | rules?\ for\ you | rules?\s*[:.] |
        forget\ (?:what|everything|about|the|that) | disregard | ignore\ what |
        must\ (?:be|always|never|use|follow|include|return|run|go|have|stay) | has\ to\ be | have\ to\ be | should\ (?:always|never) | (?:must|should)\ not |
        no\ longer | no\ more | only\ use | stop\ (?:doing|using|suggesting|calling|recommending) |
        remind\ me | so\ you\ (?:can\ )?remember | for\ (?:the|your)\ record | note\ (?:this|that)\ down |
        allerg(?:y|ic|ies) | vegetarian | vegan | gluten[- ]free | lactose | kosher | halal
    )\b""",
    re.IGNORECASE,
)
RULE_RE_CJK = re.compile(
    r"以后|今后|从今以后|从现在开始|别再|不要再|不要|永远不|一定要|务必|记住|请不要|请一定|每次|规则|只做|只用|"
    r"我喜欢|我不喜欢|我讨厌|我偏好|过敏|素食|不吃|叫我|称呼我|忘掉|忘记我|不用再"
)
CHUNK_ORDINAL_RE = re.compile(r"(?:chunk|part|seg|msg)[-_:](\d+)", re.IGNORECASE)
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)   # epoch + timedelta renders pre-1970 stamps on every platform
POS_STRIDE = 1000  # positions inside one Add chunk: ordinal * POS_STRIDE + message index


class ConflictingRequestError(Exception):
    """request_id reused with a different payload."""


@dataclass(frozen=True)
class Chunk:
    id: str
    user_id: str
    session_id: str
    role: str
    content: str
    event_ms: Optional[int]   # message timestamp (unix ms) if the adapter sent one
    pos: int                  # conversation position (ordinal * POS_STRIDE + index)
    standing: bool            # tagged as a standing rule / preference at Add time
    stored_at: str

    @property
    def order_key(self) -> tuple[int, int]:
        """Larger = newer. Falls back to the Add position when no timestamp was sent."""
        return (self.event_ms or 0, self.pos)


def cjk_only(text: str) -> str:
    """CJK characters of a text with everything else removed, so trigram matching ignores punctuation,
    spaces and interleaved Latin tokens (those are served by the word index)."""
    return "".join(CJK_RE.findall(text))


def render(chunk: Chunk, time_notes: bool = True, turn_marks: bool = False) -> str:
    """`[stamp] ROLE: text` plus, when the turn has a timestamp and relative-time phrases, a resolved-dates note (v0.4).
    A turn that came without a timestamp can carry its place in the conversation instead (`[turn 1203]`, v0.8
    TURN_MARKS): results are returned in relevance order, so without it nothing tells the reader which of two
    untimed lines was said first."""
    stamp = ""
    if chunk.event_ms is not None:
        try:
            stamp = "[" + (EPOCH + timedelta(milliseconds=chunk.event_ms)).strftime("%Y-%m-%d %H:%M UTC") + "] "
        except (OverflowError, OSError, ValueError):
            stamp = ""
    mark = f"[turn {chunk.pos + 1}] " if turn_marks and not stamp else ""
    text = f"{stamp}{mark}{chunk.role.upper()}: {chunk.content}"
    if time_notes and stamp:
        text += notes_suffix(chunk.content, chunk.event_ms)
    return text


def looks_like_rule(role: str, text: str) -> bool:
    if role != "user":
        return False
    if RULE_RE.search(text):
        return True
    return CJK_RE.search(text) is not None and RULE_RE_CJK.search(text) is not None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_log (
    request_id  TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    rowid      INTEGER PRIMARY KEY,
    id         TEXT UNIQUE NOT NULL,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    event_ms   INTEGER,
    pos        INTEGER NOT NULL,
    standing   INTEGER NOT NULL DEFAULT 0,
    stored_at  TEXT NOT NULL,
    vec        BLOB
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_mem_order    ON memories(session_id, pos);
CREATE INDEX IF NOT EXISTS ix_mem_standing ON memories(standing, event_ms);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, content='memories', content_rowid='rowid', tokenize = 'porter unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts_tri USING fts5(content, mem_id UNINDEXED, tokenize = 'trigram');
"""


def user_db_name(user_id: str) -> str:
    """Stable file name for a user: a hash, so arbitrary user_id strings never touch the file system."""
    return hashlib.blake2b(user_id.encode("utf-8"), digest_size=16).hexdigest()


class MemoryStore:
    """One SQLite file per user_id (``<data_dir>/users/ab/cd/<hash>.db``).

    Why: the evaluation holds thousands of users and ~10^8 tokens; a single shared FTS index makes
    every query scan posting lists of the whole corpus (0.9 s at 1/35 of the official volume in our
    soak test), while a per-user file keeps each query on a few hundred turns. It also makes the
    user-isolation rule physical: a Search can only ever open the file of the user_id it was given.
    """

    def __init__(self, data_dir: str, retention_days: int = 30, pool_size: int = 128,
                 cache_kb: int = 256, idle_seconds: int = 120) -> None:
        self.data_dir = Path(data_dir)
        self.users_dir = self.data_dir / "users"
        self.retention_days = retention_days
        self.pool_size = pool_size
        self.cache_kb = cache_kb          # per-connection page cache; memory is bounded by pool_size x cache_kb
        self.idle_seconds = idle_seconds
        # Open connections stay cached per process: opening and closing a WAL file on every request makes
        # the last close checkpoint the WAL, and 16 concurrent requests on one user then spend hundreds of
        # milliseconds in SQLite's busy handler (measured with ab: 765 ms p50 at concurrency 16 vs 5 ms alone).
        self._pool: "OrderedDict[str, tuple[sqlite3.Connection, threading.Lock]]" = OrderedDict()
        self._last_used: dict[str, float] = {}
        self._pool_lock = threading.Lock()
        self._checkouts = 0
        # v0.5 dense channel: per-user vector matrices cached in memory (LRU), dropped on every write to that user
        self.vector_cache_users = 64
        self._vectors: "OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]" = OrderedDict()
        self._vectors_lock = threading.Lock()
        self.vector_failures: set[str] = set()   # users with turns stored without a vector (embedding call failed)
        self.initialize()

    def initialize(self) -> None:
        self.users_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        name = user_db_name(user_id)
        return self.users_dir / name[:2] / name[2:4] / f"{name}.db"

    def _checkout(self, path: Path, create: bool) -> Optional[tuple[sqlite3.Connection, threading.Lock]]:
        key = str(path)
        with self._pool_lock:
            self._checkouts += 1
            if self._checkouts % 256 == 0:
                self._evict_idle_locked()
            entry = self._pool.get(key)
            if entry is not None:
                self._pool.move_to_end(key)
                self._last_used[key] = time.monotonic()
                return entry
            if not create and not path.exists():
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(key, timeout=30.0, check_same_thread=False)
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 30000")
                self._prepare(conn)
            except Exception:
                conn.close()
                raise
            entry = (conn, threading.Lock())
            self._pool[key] = entry
            self._last_used[key] = time.monotonic()
            while len(self._pool) > self.pool_size:
                old_key, (old_conn, old_lock) = self._pool.popitem(last=False)
                self._last_used.pop(old_key, None)
                with old_lock:
                    old_conn.close()
            return entry

    def _evict_idle_locked(self) -> None:
        """Called with the pool lock held: close connections idle for longer than `idle_seconds`
        (closing the last connection to a file also checkpoints and removes its WAL)."""
        deadline = time.monotonic() - self.idle_seconds
        for key in [k for k, t in self._last_used.items() if t < deadline]:
            entry = self._pool.pop(key, None)
            self._last_used.pop(key, None)
            if entry is not None:
                conn, lock = entry
                with lock:
                    conn.close()

    def _evict(self, path: Path) -> None:
        with self._pool_lock:
            entry = self._pool.pop(str(path), None)
            self._last_used.pop(str(path), None)
        if entry is not None:
            conn, lock = entry
            with lock:
                conn.close()

    @contextmanager
    def connect(self, user_id: str, create: bool = True) -> Iterator[Optional[sqlite3.Connection]]:
        """The user's cached connection, held exclusively for the duration of the block. Yields None
        when the user has no data and create=False (a Search for an unknown user must not create files)."""
        entry = self._checkout(self.path_for(user_id), create)
        if entry is None:
            yield None
            return
        conn, lock = entry
        with lock:
            yield conn

    def close(self) -> None:
        with self._pool_lock:
            entries = list(self._pool.values())
            self._pool.clear()
        for conn, lock in entries:
            with lock:
                conn.close()

    def _prepare(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA cache_size = -{int(self.cache_kb)}")   # KiB; the OS page cache does the rest
        conn.execute("PRAGMA journal_size_limit = 4194304")   # WAL truncated back to <= 4 MB after checkpoints
        conn.execute("PRAGMA wal_autocheckpoint = 500")       # pages (~2 MB) between automatic checkpoints
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version == SCHEMA_VERSION:
            return
        if version != 0:
            raise RuntimeError(f"user database has schema version {version}, this code expects {SCHEMA_VERSION}; move the data directory away")
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    @staticmethod
    def _index(conn: sqlite3.Connection, rowid: int, content: str, mem_id: str) -> None:
        conn.execute("INSERT INTO memories_fts(rowid, content) VALUES (?,?)", (rowid, content))  # external content: index only
        cjk = cjk_only(content)
        if cjk:  # the trigram index only carries CJK text; English corpora pay nothing for it
            conn.execute("INSERT INTO memories_fts_tri(rowid, content, mem_id) VALUES (?,?,?)", (rowid, cjk, mem_id))

    # ---------------------------------------------------------------- add
    def seen_request(self, user_id: str, request_id: str) -> bool:
        """True when this request_id was already ingested for the user (lets the caller skip embedding a replay)."""
        with self.connect(user_id, create=False) as conn:
            if conn is None:
                return False
            return conn.execute("SELECT 1 FROM ingest_log WHERE request_id = ?", (request_id,)).fetchone() is not None

    def add(self, payload: AddRequest, vectors: Optional[list[Optional[bytes]]] = None,
            embed_identity: Optional[tuple[str, int]] = None) -> int:
        """Persist one Add call in a single transaction on the user's own database. Returns the number
        of turns stored (0 for an idempotent replay). Raises ConflictingRequestError when the
        request_id was already used with a different body. `vectors` (one float16 blob or None per message,
        aligned with payload.messages) and `embed_identity` = (model, dims) come from the dense channel."""
        canonical = json.dumps(payload.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        received_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        ordinal = CHUNK_ORDINAL_RE.search(payload.request_id)
        with self.connect(payload.user_id) as conn:
            assert conn is not None
            try:
                stored = self._write(conn, payload, fingerprint, received_at, ordinal, vectors, embed_identity)
            except BaseException:
                conn.rollback()
                raise
        if stored:
            self._drop_vectors(payload.user_id)
            if vectors is not None and any(v is None for v in vectors):
                self.vector_failures.add(payload.user_id)
        return stored

    def _write(self, conn: sqlite3.Connection, payload: AddRequest, fingerprint: str, received_at: str, ordinal,
               vectors: Optional[list[Optional[bytes]]] = None, embed_identity: Optional[tuple[str, int]] = None) -> int:
        claimed = conn.execute(
            "INSERT OR IGNORE INTO ingest_log(request_id, fingerprint, session_id, received_at) VALUES (?,?,?,?)",
            (payload.request_id, fingerprint, payload.session_id, received_at),
        ).rowcount
        if claimed == 0:  # already ingested (or being ingested by another worker): replay or conflict
            prior = conn.execute("SELECT fingerprint FROM ingest_log WHERE request_id = ?", (payload.request_id,)).fetchone()
            if prior is not None and prior["fingerprint"] != fingerprint:
                raise ConflictingRequestError(f"request_id {payload.request_id!r} was already used with a different body")
            conn.rollback()  # the ignored insert opened a write transaction; never leave it open on a cached connection
            return 0
        base = int(ordinal.group(1)) * POS_STRIDE if ordinal else self._next_block(conn, payload.session_id)
        if embed_identity is not None:
            self._check_embed_identity(conn, embed_identity)
        stored = 0
        chars = 0
        for index, message in enumerate(payload.messages):
            text = message.text().strip()
            if not text:
                continue
            mem_id = "mem_" + hashlib.blake2b(f"{payload.request_id}\x1f{index}".encode("utf-8"), digest_size=10).hexdigest()
            vec = vectors[index] if vectors is not None and index < len(vectors) else None
            row = conn.execute(
                "INSERT INTO memories(id, request_id, session_id, role, content, event_ms, pos, standing, stored_at, vec)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (mem_id, payload.request_id, payload.session_id, message.role, text,
                 message.timestamp, base + index, int(looks_like_rule(message.role, text)), received_at, vec),
            )
            self._index(conn, row.lastrowid, text, mem_id)
            stored += 1
            chars += len(text)
        if stored:
            self._bump_corpus(conn, stored, chars)
        conn.commit()
        return stored

    @staticmethod
    def _bump_corpus(conn: sqlite3.Connection, turns: int, chars: int) -> None:
        """Running totals of this user's stored turns and text length (v0.8 fit mode asks "does the whole memory
        fit into the answer window?" on every Search; a SUM over the table would read every row each time)."""
        have = {r["key"] for r in conn.execute("SELECT key FROM meta WHERE key IN ('turns', 'chars')")}
        if have != {"turns", "chars"}:   # first write, or a file from before the counters existed: count once
            r = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(content)), 0) AS c FROM memories").fetchone()
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('turns', ?), ('chars', ?)", (str(r["n"]), str(r["c"])))
            return
        conn.execute("UPDATE meta SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) WHERE key = 'turns'", (turns,))
        conn.execute("UPDATE meta SET value = CAST(CAST(value AS INTEGER) + ? AS TEXT) WHERE key = 'chars'", (chars,))

    @staticmethod
    def _next_block(conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute("SELECT MAX(pos) AS p FROM memories WHERE session_id = ?", (session_id,)).fetchone()
        last = row["p"] if row and row["p"] is not None else -1
        return (int(last) // POS_STRIDE + 1) * POS_STRIDE

    # ------------------------------------------------------------- search
    @staticmethod
    def _chunk(user_id: str, row: sqlite3.Row) -> Chunk:
        return Chunk(id=row["id"], user_id=user_id, session_id=row["session_id"], role=row["role"],
                     content=row["content"], event_ms=row["event_ms"], pos=row["pos"], standing=bool(row["standing"]),
                     stored_at=row["stored_at"])

    @contextmanager
    def _reader(self, user_id: str, conn: Optional[sqlite3.Connection]) -> Iterator[Optional[sqlite3.Connection]]:
        """Reuse an open connection when the caller holds one (a Search opens the user's file once)."""
        if conn is not None:
            yield conn
            return
        with self.connect(user_id, create=False) as fresh:
            yield fresh

    def _match(self, table: str, user_id: str, expr: str, limit: int, conn: Optional[sqlite3.Connection] = None,
               event_range: Optional[tuple[Optional[int], Optional[int]]] = None) -> list[tuple[Chunk, float]]:
        if not expr:
            return []
        sql = (f"SELECT m.*, bm25({table}) AS r FROM {table} JOIN memories m ON m.rowid = {table}.rowid"
               f" WHERE {table} MATCH ?")  # bm25() needs no stored text, so the external-content index is enough
        params: list = [expr]
        if event_range is not None:  # planner date range: only turns whose timestamp falls inside it
            lo, hi = event_range
            if lo is not None:
                sql += " AND m.event_ms >= ?"
                params.append(lo)
            if hi is not None:
                sql += " AND m.event_ms <= ?"
                params.append(hi)
        sql += " ORDER BY r LIMIT ?"
        params.append(limit)
        with self._reader(user_id, conn) as c:
            if c is None:
                return []
            try:
                rows = c.execute(sql, params).fetchall()
            except sqlite3.OperationalError:  # malformed MATCH expression -> treat as no hits
                return []
        return [(self._chunk(user_id, r), -float(r["r"])) for r in rows]  # bm25() is negative; flip so higher is better

    def fts_search(self, user_id: str, expr: str, limit: int, conn: Optional[sqlite3.Connection] = None,
                   event_range: Optional[tuple[Optional[int], Optional[int]]] = None) -> list[tuple[Chunk, float]]:
        """Word-level (porter-stemmed) BM25 search."""
        return self._match("memories_fts", user_id, expr, limit, conn, event_range)

    def trigram_search(self, user_id: str, expr: str, limit: int, conn: Optional[sqlite3.Connection] = None,
                       event_range: Optional[tuple[Optional[int], Optional[int]]] = None) -> list[tuple[Chunk, float]]:
        """Character-trigram BM25 search over the CJK text of each turn."""
        return self._match("memories_fts_tri", user_id, expr, limit, conn, event_range)

    def span(self, chunk: Chunk, window: int, conn: Optional[sqlite3.Connection] = None) -> list[Chunk]:
        """The turn plus `window` turns before and after it, in conversation order. Crosses Add-chunk
        boundaries (pos jumps by POS_STRIDE there, so a BETWEEN would stop short)."""
        if window <= 0:
            return [chunk]
        with self._reader(chunk.user_id, conn) as c:
            if c is None:
                return [chunk]
            before = c.execute("SELECT * FROM memories WHERE session_id = ? AND pos < ? ORDER BY pos DESC LIMIT ?",
                               (chunk.session_id, chunk.pos, window)).fetchall()
            after = c.execute("SELECT * FROM memories WHERE session_id = ? AND pos > ? ORDER BY pos LIMIT ?",
                              (chunk.session_id, chunk.pos, window)).fetchall()
        u = chunk.user_id
        return [self._chunk(u, r) for r in reversed(before)] + [chunk] + [self._chunk(u, r) for r in after]

    def corpus_size(self, user_id: str, conn: Optional[sqlite3.Connection] = None) -> tuple[int, int]:
        """(turns, characters of stored text) for this user."""
        with self._reader(user_id, conn) as c:
            if c is None:
                return (0, 0)
            rows = {r["key"]: r["value"] for r in c.execute("SELECT key, value FROM meta WHERE key IN ('turns', 'chars')")}
            if "turns" in rows and "chars" in rows:
                return (int(rows["turns"]), int(rows["chars"]))
            r = c.execute("SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(content)), 0) AS c FROM memories").fetchone()
            return (int(r["n"]), int(r["c"]))

    def all_turns(self, user_id: str, conn: Optional[sqlite3.Connection] = None) -> list[Chunk]:
        """Every stored turn in conversation order: sessions in the order they first appeared, turns by position."""
        with self._reader(user_id, conn) as c:
            if c is None:
                return []
            rows = c.execute("SELECT id, session_id, role, content, event_ms, pos, standing, stored_at"
                             " FROM memories ORDER BY rowid").fetchall()
        first_seen: dict[str, int] = {}
        for r in rows:
            first_seen.setdefault(r["session_id"], len(first_seen))
        rows.sort(key=lambda r: (first_seen[r["session_id"]], r["pos"]))
        return [self._chunk(user_id, r) for r in rows]

    def standing_rules(self, user_id: str, limit: int, conn: Optional[sqlite3.Connection] = None) -> list[Chunk]:
        if limit <= 0:
            return []
        with self._reader(user_id, conn) as c:
            if c is None:
                return []
            rows = c.execute("SELECT * FROM memories WHERE standing = 1 ORDER BY event_ms DESC, pos DESC LIMIT ?", (limit,)).fetchall()
        return [self._chunk(user_id, r) for r in rows]

    def newest(self, user_id: str, limit: int, conn: Optional[sqlite3.Connection] = None) -> list[Chunk]:
        if limit <= 0:
            return []
        with self._reader(user_id, conn) as c:
            if c is None:
                return []
            rows = c.execute("SELECT * FROM memories ORDER BY event_ms DESC, pos DESC LIMIT ?", (limit,)).fetchall()
        return [self._chunk(user_id, r) for r in rows]

    def count(self, user_id: str) -> int:
        with self.connect(user_id, create=False) as conn:
            if conn is None:
                return 0
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    # ------------------------------------------------------------- dense channel (v0.5)
    @staticmethod
    def _check_embed_identity(conn: sqlite3.Connection, identity: tuple[str, int]) -> None:
        """A user file records the first embedding model/dims written into it and refuses any other,
        so vectors from two models can never be mixed in one matrix."""
        model, dims = identity
        row = conn.execute("SELECT value FROM meta WHERE key = 'embed'").fetchone()
        wanted = f"{model}:{dims}"
        if row is None:
            conn.execute("INSERT INTO meta(key, value) VALUES ('embed', ?)", (wanted,))
        elif row["value"] != wanted:
            raise RuntimeError(f"user database holds {row['value']} vectors, this process embeds with {wanted}; move the data directory away")

    def _drop_vectors(self, user_id: str) -> None:
        with self._vectors_lock:
            self._vectors.pop(user_id, None)

    def _matrix(self, user_id: str, dims: int, conn: sqlite3.Connection):
        """(matrix float32 [n, dims], rowids int64, event_ms float64 (nan = none), ids) for every turn with a vector."""
        with self._vectors_lock:
            cached = self._vectors.get(user_id)
            if cached is not None:
                self._vectors.move_to_end(user_id)
                return cached
        rows = conn.execute("SELECT rowid, id, event_ms, vec FROM memories WHERE vec IS NOT NULL ORDER BY rowid").fetchall()
        if rows:
            matrix = np.vstack([from_blob(r["vec"], dims) for r in rows])
            rowids = np.asarray([r["rowid"] for r in rows], dtype=np.int64)
            event = np.asarray([r["event_ms"] if r["event_ms"] is not None else np.nan for r in rows], dtype=np.float64)
            ids = [r["id"] for r in rows]
        else:
            matrix = np.zeros((0, dims), dtype=np.float32); rowids = np.zeros(0, dtype=np.int64)
            event = np.zeros(0, dtype=np.float64); ids = []
        entry = (matrix, rowids, event, ids)
        with self._vectors_lock:
            self._vectors[user_id] = entry
            while len(self._vectors) > self.vector_cache_users:
                self._vectors.popitem(last=False)
        return entry

    def dense_search(self, user_id: str, qvec: np.ndarray, limit: int, conn: Optional[sqlite3.Connection] = None,
                     event_range: Optional[tuple[Optional[int], Optional[int]]] = None) -> list[tuple[Chunk, float]]:
        """Cosine top-`limit` over the user's stored vectors (brute force; a user holds a few hundred to a few
        thousand turns, so a dot product is faster than any index). Scores are cosine similarities in [-1, 1]."""
        if limit <= 0:
            return []
        with self._reader(user_id, conn) as c:
            if c is None:
                return []
            matrix, rowids, event, _ = self._matrix(user_id, int(qvec.shape[0]), c)
            if matrix.shape[0] == 0:
                return []
            scores = matrix @ qvec.astype(np.float32)
            if event_range is not None:
                lo, hi = event_range
                mask = ~np.isnan(event)
                if lo is not None:
                    mask &= event >= lo
                if hi is not None:
                    mask &= event <= hi
                scores = np.where(mask, scores, -np.inf)
            k = min(limit, scores.shape[0])
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            top = [int(i) for i in top if np.isfinite(scores[i])]
            if not top:
                return []
            wanted = [int(rowids[i]) for i in top]
            marks = ",".join("?" * len(wanted))
            rows = {r["rowid"]: r for r in c.execute(f"SELECT rowid, * FROM memories WHERE rowid IN ({marks})", wanted).fetchall()}
        return [(self._chunk(user_id, rows[rid]), float(scores[i])) for i, rid in zip(top, wanted) if rid in rows]

    def missing_vectors(self, user_id: str, limit: int) -> list[tuple[int, str]]:
        """(rowid, content) of turns stored without a vector, oldest first."""
        with self.connect(user_id, create=False) as conn:
            if conn is None:
                return []
            rows = conn.execute("SELECT rowid, content FROM memories WHERE vec IS NULL ORDER BY rowid LIMIT ?", (limit,)).fetchall()
            return [(int(r["rowid"]), r["content"]) for r in rows]

    def store_vectors(self, user_id: str, pairs: list[tuple[int, bytes]]) -> int:
        if not pairs:
            return 0
        with self.connect(user_id, create=False) as conn:
            if conn is None:
                return 0
            conn.executemany("UPDATE memories SET vec = ? WHERE rowid = ? AND vec IS NULL", [(b, r) for r, b in pairs])
            conn.commit()
        self._drop_vectors(user_id)
        return len(pairs)

    def backfill_vectors(self, embedder: Embedder, limit_per_user: int = 200) -> int:
        """Embed turns that were stored without a vector (service failure at Add time). Returns vectors written."""
        written = 0
        for user_id in list(self.vector_failures):
            todo = self.missing_vectors(user_id, limit_per_user)
            if not todo:
                self.vector_failures.discard(user_id)
                continue
            vecs = embedder.embed([t for _, t in todo])
            pairs = [(rid, to_blob(v)) for (rid, _), v in zip(todo, vecs) if v is not None]
            written += self.store_vectors(user_id, pairs)
            if len(pairs) == len(todo) and len(todo) < limit_per_user:
                self.vector_failures.discard(user_id)
        return written

    # --------------------------------------------------------- governance
    def delete_expired(self) -> int:
        """Delete every user database whose last write is older than `retention_days`
        (competition rule: evaluation data must be gone within 30 days of the run). Returns files removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).timestamp()
        removed = 0
        for db in self.users_dir.glob("*/*/*.db"):
            try:
                if db.stat().st_mtime >= cutoff:
                    continue
                self._evict(db)
                for suffix in ("", "-wal", "-shm"):
                    extra = Path(str(db) + suffix)
                    if extra.exists():
                        extra.unlink()
                removed += 1
            except OSError:
                continue
        return removed
