# Multi-Target Production Dockerfile for Fraud Intelligence Platform
# Supports both FastAPI Microservice and Streamlit Analytics Dashboard
# Author: Manas Jain <manasj1234567890@gmail.com>

FROM python:3.11-slim as base

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install security updates and curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_full.txt .
RUN pip install --no-cache-dir -r requirements_full.txt

# Copy application assets
COPY api.py app.py models_engine.py paysim_engine.py data_generator.py ./
COPY data/ ./data/
COPY simulations/ ./simulations/
COPY paysim_benchmark/ ./paysim_benchmark/
COPY tests/ ./tests/

# Create non-root system user for enterprise security best practices
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Healthcheck probing FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Default entrypoint starts the FastAPI microservice
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
