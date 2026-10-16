FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0 \
    MPLCONFIGDIR=/tmp/matplotlib

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

COPY . .

# The app writes each job's silhouettes, measurements, and viewer under runs/.
RUN mkdir -p /app/runs /tmp/matplotlib \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app/runs /tmp/matplotlib

USER appuser

EXPOSE 5000

# One Gunicorn worker is intentional: job state is currently held in memory.
# Threads still allow concurrent requests while pipeline jobs run as subprocesses.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads ${WEB_CONCURRENCY_THREADS:-8} --timeout 180 app:app"]
