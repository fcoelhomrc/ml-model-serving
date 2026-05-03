FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra inference

COPY main.py ./

EXPOSE 8080
CMD ["uv", "run", "--extra", "inference", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
