# ─────────────────────────────────────────────────────────────────────────
#  DHS Stunting Risk Prediction Service — Dockerfile
#  DSA 8401 Applied Machine Learning Capstone
# ─────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="DSA 8401 Capstone"
LABEL description="Stunting risk prediction API using Kenya 2022 DHS data"


# Working directory
WORKDIR /app

# Install Python dependencies first (cache-friendly layer ordering)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY static/ ./static/

ENV PORT=8000
EXPOSE 8000

# Run the API server
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

