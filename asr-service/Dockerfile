# in ./asr-service/Dockerfile

FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install dependencies including ffmpeg and python3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- FIX: Be explicit about using the python3.11 pip ---
# Upgrade pip for python3.11 specifically
RUN python3.11 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app
COPY . .

# --- FIX: Use python3.11's pip and enable the cache ---
# This ensures packages are installed for the correct Python version and uses your cache volume
RUN python3.11 -m pip install --cache-dir /cache \
    fastapi \
    uvicorn \
    torch --extra-index-url https://download.pytorch.org/whl/cu124 \
    openai-whisper \
    numpy \
    soundfile \
    python-multipart \
    requests

EXPOSE 9000

# --- FIX: Run the application with python3.11 ---
CMD ["python3.11", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "9000"]
