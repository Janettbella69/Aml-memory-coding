FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/srv/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY scripts ./scripts
ENV PATH="/srv/.venv/bin:$PATH" DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4)" || exit 1
# shell form so UVICORN_WORKERS from .env applies; per-user SQLite files make multi-process safe
CMD uvicorn --factory app.main:create_app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}
