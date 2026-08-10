FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout 300 --retries 10 -r requirements.txt

COPY . .

# Defensive fallback: if the container is ever run without an explicit -v mount for
# /app/knowledge (embeddings cache + chat logs), Docker creates an anonymous volume
# here instead of storing that data only in the writable layer, where it would be
# silently lost on container removal.
VOLUME ["/app/knowledge"]

EXPOSE 8000

# --limit-concurrency raised well above PROCESSING_SLOTS (60) + MAX_QUEUE_DEPTH (40) in main.py,
# so uvicorn never rejects a connection that the app-level queue would otherwise have handled.
# It stays as a broad backstop against runaway connection counts, just far above the app's own ceiling.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "200"]
