FROM python:3.11-slim

WORKDIR /app

# Apply latest system cybersecurity patches and install minimal dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Security: Create an unprivileged user for executing the application
RUN groupadd -r pneumoops && useradd -r -g pneumoops appuser

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and ensure secure permissions
COPY --chown=appuser:pneumoops backend  ./backend
COPY --chown=appuser:pneumoops frontend ./frontend
COPY --chown=appuser:pneumoops scripts  ./scripts
COPY --chown=appuser:pneumoops models   ./models
COPY --chown=appuser:pneumoops data     ./data
COPY --chown=appuser:pneumoops model_utils.py ./model_utils.py
COPY --chown=appuser:pneumoops README.md ./README.md

ENV PYTHONPATH=/app
ENV PORT=7860
ENV MPLCONFIGDIR=/tmp/matplotlib
# Default to the ChestMNIST + MobileNetV3-small profile (assignment target)
ENV PNEUMOOPS_PROFILE=chestmnist

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/health || exit 1

# Security: Drop root privileges before running the application
USER appuser

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
