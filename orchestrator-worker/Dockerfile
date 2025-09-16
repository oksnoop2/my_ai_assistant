# my_ai_assistant/orchestrator/Dockerfile (Correct Version)

FROM python:3.11-slim

# We only need audio libraries now.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libportaudio2 \
    libspa-0.2-jack \
    pipewire-alsa \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# REMOVED: The 'docker' library is no longer needed.
RUN pip3 install --no-cache-dir requests sounddevice scipy numpy

CMD ["python3", "orchestrator.py"]
