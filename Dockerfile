FROM python:3.11-slim

WORKDIR /app

# Native libs needed by scientific Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend  ./backend
COPY frontend ./frontend
COPY scripts  ./scripts
COPY models   ./models
COPY data     ./data
COPY model_utils.py ./model_utils.py
COPY README.md ./README.md

ENV PYTHONPATH=/app
ENV PORT=7860
# Default to the ChestMNIST + MobileNetV3-small profile (assignment target)
ENV PNEUMOOPS_PROFILE=chestmnist

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
