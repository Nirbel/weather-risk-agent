# One image for both services. docker-compose.yml runs it twice: `backend` (FastAPI) and
# `frontend` (Streamlit). Run on its own, the image starts both with scripts/run.py.
# The default way to run the app is still `uv run python scripts/run.py` on the host.
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

# Dependencies first (cached layer), installed from the hashed lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data && chown -R app:app /app/data
USER app

EXPOSE 8000 8501
# Data is fetched on demand and cached in /app/data/app.db; mount a volume there to keep it.
CMD ["python", "scripts/run.py", "--host", "0.0.0.0"]
