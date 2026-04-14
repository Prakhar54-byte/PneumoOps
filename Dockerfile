FROM python:3.11-slim

WORKDIR /app

# Keep the base image lean but include the minimum native packages commonly needed by scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY models ./models
COPY data ./data
COPY README.md ./README.md
COPY docs ./docs

ENV PYTHONPATH=/app
ENV PORT=7860
ENV PNEUMOOPS_PROFILE=prototype

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
