FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r requirements.txt 2>/dev/null || \
    pip install --no-cache-dir openenv-core>=0.2.0 fastapi uvicorn[standard] \
    pydantic pandas numpy openai python-multipart pyyaml httpx scipy scikit-learn

COPY . .

RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
