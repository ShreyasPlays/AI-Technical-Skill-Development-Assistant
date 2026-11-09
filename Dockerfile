# Multi-stage production Dockerfile for AI Technical Skill Development Assistant
# Built on official Python 3.11-slim base image for minimal attack surface and fast startup.

FROM python:3.11-slim as runtime

# Set environment variables for Python in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install lightweight OS packages required for healthchecks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-privileged user and group for security
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -s /bin/bash -m appuser

# Install pinned dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained model artifacts
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/
COPY reports/ /app/reports/

# Grant appropriate permissions to appuser
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose microservice REST port
EXPOSE 8000

# Container liveness & readiness healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server via Uvicorn
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

