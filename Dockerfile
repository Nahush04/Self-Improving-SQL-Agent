FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e .

# The embedding model downloads lazily on first use and is cached in this volume
# alongside the SQLite file, so it only needs internet access once, not baked into
# the image (keeps the build itself lighter).
VOLUME ["/data", "/root/.cache/huggingface"]
ENV SQLAGENT_MEMORY_DB=/data/memory.sqlite

EXPOSE 8000

ENTRYPOINT ["python", "-m", "sqlagent.memory.server"]
CMD ["--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
