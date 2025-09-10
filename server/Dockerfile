# server/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Non-root user for Cloud Run
RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Install deps first (better layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application (includes ./models/** unless excluded by .dockerignore)
COPY . .

# Debug: show model files in build logs
RUN set -eux; ls -alh ./models || true; ls -alh ./models/sentiment || true

# Drop privileges
USER appuser

# Cloud Run env & defaults
ENV PORT=8080 \
    API_VERSION=2.1.2 \
    CORS_ALLOWED_ORIGINS=*

EXPOSE 8080

# Use shell form so ${PORT} expands; 'exec' preserves PID 1 signals
CMD ["bash", "-lc", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --log-level info"]
