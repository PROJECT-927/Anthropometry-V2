FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=5000 \
    HOST=0.0.0.0 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    WEB_CONCURRENCY_THREADS=8 \
    MAX_CONCURRENT_COMPUTE_JOBS=1 \
    MAX_PENDING_COMPUTE_JOBS=8 \
    MAX_GRID_VOXELS=400000000

WORKDIR /app

# OpenCV and MediaPipe use these shared libraries in the CPU-only container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy the current Python modules (including compute_queue, voxel_memory and
# memory_diagnostics) plus the browser assets and pinned pose model. Rebuilding
# the image therefore includes the optimized pipeline, without local run data.
COPY *.py hull_viewer_template.html pose_landmarker_heavy.task ./
COPY templates/ ./templates/
COPY static/ ./static/

# The app writes each job's silhouettes, measurements, and viewer under runs/.
RUN mkdir -p /app/runs /tmp/matplotlib \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app/runs /tmp/matplotlib

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '5000') + '/', timeout=4).close()"]

# One Gunicorn worker is intentional: job state is currently held in memory.
# Request threads keep the UI responsive; the shared compute queue separately
# limits pipeline/viewer/STL work to MAX_CONCURRENT_COMPUTE_JOBS (default one).
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads ${WEB_CONCURRENCY_THREADS:-8} --timeout 180 --error-logfile - --capture-output app:app"]
