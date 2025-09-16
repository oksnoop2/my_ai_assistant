FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# --- THIS IS THE FIX ---
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    openai-whisper \
    numpy \
    soundfile \
    python-multipart

EXPOSE 9000
CMD ["uvicorn", "asr:app", "--host", "0.0.0.0", "--port", "9000"]
