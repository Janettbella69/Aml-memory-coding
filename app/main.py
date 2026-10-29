from __future__ import annotations

import asyncio
import logging
import secrets
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

from . import __version__
from .config import Settings
from .embed import Embedder, embedder_from_env, to_blob
from .models import AddRequest, AddResponse, HealthResponse, SearchRequest, SearchResponse
from .planner import Planner, planner_from_env
from .rerank import Reranker, reranker_from_env
from .selector import Selector, selector_from_env
from .search import search as assemble_evidence
from .store import ConflictingRequestError, MemoryStore

log = logging.getLogger("aml_memory")


def create_app(settings: Settings | None = None, store: MemoryStore | None = None,
               planner: Planner | None = None, embedder: Embedder | None = None,
               selector: Selector | None = None, reranker: Reranker | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    logging.basicConfig(level=cfg.log_level)
    memory = store or MemoryStore(cfg.data_dir, retention_days=cfg.retention_days, pool_size=cfg.db_pool_size,
                                  cache_kb=cfg.db_cache_kb, idle_seconds=cfg.db_idle_seconds)
    # SQLite serialises allocations behind one process-wide mutex; 16 request threads hammering it spend more
    # time contending than querying (measured: 24 q/s at 412 % CPU). A small gate per process restores throughput.
    search_gate = threading.BoundedSemaphore(cfg.search_parallelism)
    write_gate = threading.BoundedSemaphore(cfg.write_parallelism)
    query_planner = planner or planner_from_env(cfg.planner_model, cfg.planner_max_queries, cfg.planner_timeout)
    if query_planner is not None:
        log.info("query planner enabled: %s", cfg.planner_model)
    evidence_selector = selector or selector_from_env(cfg.selector_model, cfg.selector_timeout,
                                                      cfg.selector_max_select, cfg.selector_block_chars)
    if evidence_selector is not None:
        log.info("evidence selector enabled: %s (candidates %d, tail %d)", cfg.selector_model,
                 cfg.selector_candidates, cfg.selector_tail)
    cross_reranker = reranker or reranker_from_env(cfg.rerank_provider, cfg.rerank_model, cfg.rerank_timeout, cfg.rerank_doc_chars)
    if cross_reranker is not None:
        log.info("reranker enabled: %s/%s (candidates %d, mode %s)", cfg.rerank_provider, cfg.rerank_model, cfg.rerank_candidates, cfg.rerank_mode)
    dense = embedder or embedder_from_env(cfg.embed_model, cfg.embed_dims, cfg.embed_timeout, cfg.embed_rps)
    memory.vector_cache_users = cfg.vector_cache_users
    if dense is not None:
        log.info("dense channel enabled: %s (%d dims)", dense.model, dense.dims)

    async def retention_loop() -> None:
        while True:
            try:
                removed = await asyncio.to_thread(memory.delete_expired)
                if removed:
                    log.info("retention sweep removed %d user databases older than %d days", removed, cfg.retention_days)
            except Exception:  # never let the sweep take the service down
                log.exception("retention sweep failed")
            await asyncio.sleep(cfg.sweep_hours * 3600)

    async def backfill_loop() -> None:
        """Turns stored without a vector (embedding service failure at Add time) get one later."""
        while True:
            await asyncio.sleep(60)
            if dense is None or not memory.vector_failures:
                continue
            try:
                written = await asyncio.to_thread(memory.backfill_vectors, dense)
                if written:
                    log.info("vector backfill wrote %d vectors", written)
            except Exception:
                log.exception("vector backfill failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        tasks = [asyncio.create_task(retention_loop()), asyncio.create_task(backfill_loop())]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            memory.close()

    app = FastAPI(title="aml-memory", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = cfg
    app.state.store = memory

    def authenticate(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    ) -> None:
        if not cfg.api_key:
            return
        candidate = x_api_key or ""
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() in ("bearer", "token"):
                candidate = token.strip()
        if not candidate or not secrets.compare_digest(candidate.encode(), cfg.api_key.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid API key (X-Api-Key / Bearer / Token)")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    def add_memory(payload: AddRequest, _: None = Depends(authenticate)) -> AddResponse:
        vectors = None
        identity = None
        if dense is not None and not memory.seen_request(payload.user_id, payload.request_id):
            # network call outside the write gate; a replay of a known request_id skips it
            vectors = [to_blob(v) if v is not None else None for v in dense.embed([m.text() for m in payload.messages])]
            identity = (dense.model, dense.dims)
        try:
            with write_gate:
                memory.add(payload, vectors, identity)
        except ConflictingRequestError as conflict:
            raise HTTPException(status_code=409, detail=str(conflict)) from conflict
        return AddResponse(request_id=payload.request_id, user_id=payload.user_id, session_id=payload.session_id)

    def search_memory(payload: SearchRequest, _: None = Depends(authenticate)) -> SearchResponse:
        data = assemble_evidence(memory, cfg, payload.query, payload.options, payload.user_id, payload.top_k,
                                 planner=query_planner, gate=search_gate, embedder=dense,
                                 selector=evidence_selector, reranker=cross_reranker)
        log.info("search results=%d top_k=%d", len(data), payload.top_k)  # never the query or memory text
        return SearchResponse(data=data)

    for path in ("/v1/memories/add", "/add"):
        app.add_api_route(path, add_memory, methods=["POST"], response_model=AddResponse)
    for path in ("/v1/memories/search", "/search"):
        # unknown optional fields are left out, never sent as null (the platform validates every returned item)
        app.add_api_route(path, search_memory, methods=["POST"], response_model=SearchResponse, response_model_exclude_none=True)
    return app

