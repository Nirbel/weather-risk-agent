# One image, one command: the same scripts/run.py starts the API and the UI.
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Dependencies first (cached layer), installed from the hashed lockfile.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000 8501
# The database is seeded from data/seed/ on first start; mount /app/data to keep it.
CMD ["uv", "run", "--no-dev", "python", "scripts/run.py", "--host", "0.0.0.0"]
