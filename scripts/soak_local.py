"""Load test against a running server: synthetic conversations, 16-way concurrent Add then Search.

    uv run uvicorn --factory app.main:create_app --port 8000 &      # or the deployed URL
    uv run python scripts/soak_local.py --base http://127.0.0.1:8000 --users 300 --chunks 30

Prints Add throughput and latency percentiles, Search latency percentiles at the end (when the
corpus is largest), and the DB size if --db is given. Text is synthetic (Zipf-distributed vocabulary
with names/numbers/dates sprinkled in) so posting lists look like real prose; nothing here is
evaluation data.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import statistics
import threading
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

VOCAB_SIZE = 20000
NAMES = ["Mia", "Noor", "Jonas", "Lin", "Tao", "Priya", "Sam", "Ravi", "Meera", "Dani", "Wei", "Luna", "Pixel", "Biscuit"]
CITIES = ["Lisbon", "Berlin", "Hangzhou", "Shanghai", "Chengdu", "Pune", "Toronto", "Porto", "Shenzhen"]


def make_vocab(seed: int = 7) -> list[str]:
    rnd = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = set()
    while len(words) < VOCAB_SIZE:
        n = rnd.randint(3, 9)
        words.add("".join(rnd.choice(letters) for _ in range(n)))
    return sorted(words)


def zipf_sentence(rnd: random.Random, vocab: list[str], n_words: int) -> str:
    out = []
    for _ in range(n_words):
        r = rnd.random()
        if r < 0.03:
            out.append(rnd.choice(NAMES))
        elif r < 0.05:
            out.append(rnd.choice(CITIES))
        elif r < 0.07:
            out.append(str(rnd.randint(1, 3000)))
        else:
            # Zipf: index ~ 1/rank
            idx = int(len(vocab) * (rnd.random() ** 3))
            out.append(vocab[idx])
    s = " ".join(out)
    return s[0].upper() + s[1:] + "."


def _search_worker(job: tuple) -> tuple[list[float], list[int]]:
    base, headers, run, users, seed, n = job
    vocab = make_vocab()
    parsed = urllib.parse.urlsplit(base)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = cls(parsed.hostname, parsed.port, timeout=1800)
    rnd = random.Random(1000 + seed)
    lat: list[float] = []
    sizes: list[int] = []
    for _ in range(n):
        u = rnd.randrange(users)
        q = zipf_sentence(rnd, vocab, 9) + " " + rnd.choice(NAMES) + " " + rnd.choice(CITIES)
        body = json.dumps({"query": q, "user_id": f"{run}:user-{u}", "top_k": 100}).encode()
        t0 = time.perf_counter()
        conn.request("POST", "/v1/memories/search", body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        lat.append(time.perf_counter() - t0)
        if resp.status != 200:
            raise RuntimeError(f"search -> {resp.status}: {data[:200]!r}")
        sizes.append(sum(len(d["content"]) for d in json.loads(data)["data"]))
    conn.close()
    return lat, sizes


def percentiles(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    xs = sorted(xs)
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    return f"p50 {p(0.5)*1000:.0f} ms · p90 {p(0.9)*1000:.0f} ms · p99 {p(0.99)*1000:.0f} ms · max {xs[-1]*1000:.0f} ms"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=os.environ.get("MEMORY_API_KEY") or None)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=30, help="Add calls per user (20 messages each)")
    ap.add_argument("--words", type=int, default=35, help="words per message")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--searches", type=int, default=300)
    ap.add_argument("--data-dir", help="the server's DATA_DIR, to report its size on disk")
    ap.add_argument("--run", default=f"soak{int(time.time())}")
    args = ap.parse_args()

    vocab = make_vocab()
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-Api-Key"] = args.api_key

    parsed = urllib.parse.urlsplit(args.base)
    local = threading.local()

    def connection() -> http.client.HTTPConnection:
        conn = getattr(local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
            conn = local.conn = cls(parsed.hostname, parsed.port, timeout=1800)
        return conn

    def post(path: str, body: dict) -> dict:
        """One persistent keep-alive connection per worker thread (a new socket per request exhausts
        ephemeral ports on macOS after a few thousand calls)."""
        payload = json.dumps(body).encode()
        for attempt in range(3):
            conn = connection()
            try:
                conn.request("POST", path, body=payload, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"{path} -> {resp.status}: {data[:200]!r}")
                return json.loads(data)
            except (http.client.HTTPException, ConnectionError, OSError):
                local.conn = None
                if attempt == 2:
                    raise

    add_lat: list[float] = []
    lock = threading.Lock()
    errors = 0

    def add_user(u: int) -> None:
        nonlocal errors
        rnd = random.Random(u)
        uid = f"{args.run}:user-{u}"
        base_ts = 1_700_000_000_000 + u * 86_400_000
        for c in range(args.chunks):
            msgs = [{"role": "user" if i % 2 == 0 else "assistant",
                     "timestamp": base_ts + (c * 20 + i) * 60_000,
                     "content": zipf_sentence(rnd, vocab, args.words)} for i in range(20)]
            body = {"request_id": f"{args.run}:conv-{u}:chunk-{c}", "user_id": uid, "session_id": f"{args.run}:sample:{u}", "messages": msgs}
            t0 = time.perf_counter()
            try:
                post("/v1/memories/add", body)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors += 1
                print("add error:", exc)
                continue
            with lock:
                add_lat.append(time.perf_counter() - t0)

    total_msgs = args.users * args.chunks * 20
    print(f"Add: {args.users} users × {args.chunks} chunks × 20 msgs × ~{args.words} words ≈ {total_msgs*args.words/1e6:.1f} M words, concurrency {args.concurrency}")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(add_user, range(args.users)))
    dt = time.perf_counter() - t0
    print(f"  {len(add_lat)} adds in {dt:.1f} s → {len(add_lat)/dt:.1f} adds/s, {len(add_lat)*20/dt:.0f} msgs/s, errors {errors}")
    print(f"  add latency: {percentiles(add_lat)}")

    # Search phase in worker PROCESSES: a threaded Python client caps out near 20 q/s on its own GIL
    # (parsing 16 × 140 KB responses), which hid a 300 q/s server behind client overhead.
    jobs = [(args.base, headers, args.run, args.users, i, args.searches // args.concurrency) for i in range(args.concurrency)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.concurrency) as pool:
        parts = list(pool.map(_search_worker, jobs))
    dt = time.perf_counter() - t0
    search_lat = [x for lat, _ in parts for x in lat]
    sizes = [x for _, sz in parts for x in sz]
    print(f"Search: {len(search_lat)} queries, {args.concurrency} client processes, {len(search_lat)/dt:.1f} q/s")
    print(f"  search latency: {percentiles(search_lat)}")
    print(f"  response size: median {statistics.median(sizes)/1000:.0f} k chars, max {max(sizes)/1000:.0f} k chars")
    if args.data_dir and os.path.isdir(args.data_dir):
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(args.data_dir) for f in fs)
        print(f"  data dir: {size/1e6:.0f} MB on disk for ~{total_msgs*args.words*6/1e6:.0f} MB of text")


if __name__ == "__main__":
    main()
