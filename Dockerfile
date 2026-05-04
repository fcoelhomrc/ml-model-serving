FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra inference --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --extra inference

EXPOSE 8080
CMD ["uv", "run", "--extra", "inference", "uvicorn", "ml_model_serving.main:app", "--host", "0.0.0.0", "--port", "8080"]
