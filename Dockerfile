# Telesense backend image.
# Model weights download on first use into /data/hf-cache.
# CPU inference time depends on recording length and model downloads.

FROM python:3.11-slim-bookworm

# System deps:
# - ffmpeg: audio extraction / video reading used by moviepy, librosa, OpenCV
# - libgl1 + libglib2.0-0: OpenCV runtime
# - gosu: drop privileges after initializing the mounted volume
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so edits to app code don't bust the big layer.
COPY requirements-minimal.txt ./
COPY app/backend/requirements.txt ./app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "torch>=2.6,<2.9" "torchaudio>=2.6,<2.9" "torchvision>=0.21,<0.24" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r app/backend/requirements.txt

# App source + editable install of the telesense_ml package.
COPY pyproject.toml README.md ./
COPY src ./src
COPY app/backend ./app/backend
COPY configs ./configs
RUN pip install --no-cache-dir -e .

# The entrypoint initializes the mounted volume, then drops privileges.
RUN useradd -m -u 1000 telesense \
    && chown -R telesense:telesense /app
COPY docker-entrypoint.sh /usr/local/bin/telesense-entrypoint
RUN chmod +x /usr/local/bin/telesense-entrypoint
ENTRYPOINT ["/usr/local/bin/telesense-entrypoint"]

ENV HF_HOME=/data/hf-cache \
    TELESENSE_STORAGE_DIR=/data/storage \
    HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    PYTHONPATH=/app:/app/src

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
