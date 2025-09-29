FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install pip tooling and uv
RUN pip install --upgrade pip setuptools wheel
RUN pip install uv

COPY pyproject.toml uv.lock ./

# Creating the venv with only runtime deps (no dev)
RUN uv sync --no-dev --frozen

# Copying application code
COPY . .


RUN if [ -x ".venv/bin/python" ]; then \
      .venv/bin/python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet || true; \
    else \
      uv run python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet || true; \
    fi

EXPOSE 8080

CMD [".venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
