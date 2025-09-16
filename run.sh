#!/bin/bash
set -e

# --- 1. BUILD YOUR IMAGES ---
# All build paths now use the '-worker' suffix for consistency.
echo "Building container images..."
podman build -t localhost/asr-image ./asr-worker
podman build -t localhost/llm-service-image ./llm-worker
podman build -t localhost/tts-image ./tts-worker
podman build -t localhost/orchestrator-image ./orchestrator-worker
echo "✅ Images built."

# --- 2. PREPARE DIRECTORIES ---
# This ensures the directories we mount will exist.
mkdir -p ./volumes/asr/.cache
mkdir -p ./volumes/tts/voices
mkdir -p ./volumes/tts/model-cache
mkdir -p ./volumes/llm/gguf-models

# --- 3. RUN THE FULL SYSTEM ---
echo "Stopping and removing old containers..."
podman stop asr llm tts orchestrator &>/dev/null && podman rm asr llm tts orchestrator &>/dev/null || true
echo "✅ Old containers removed."

# --- ASR Service with the correct cache volume ---
echo "Starting ASR service..."
podman run --replace -d --name asr --network my-ai-network \
  -v ./volumes/asr/.cache:/root/.cache:z \
  localhost/asr-image

# --- LLM Service with the correct model volume ---
echo "Starting LLM service..."
podman run --replace -d --name llm --network my-ai-network --gpus all \
  -v ./volumes/llm/gguf-models:/models:z \
  localhost/llm-service-image

# --- TTS Service with the correct model and voice volumes ---
echo "Starting TTS service..."
podman run --replace -d --name tts --network my-ai-network --gpus all \
  -v ./volumes/tts/voices:/voices:z \
  -v ./volumes/tts/model-cache:/root/.local/share/tts:z \
  -e COQUI_TOS_AGREED=1 \
  localhost/tts-image

# --- Orchestrator Service ---
echo "✅ All services started. Launching Orchestrator..."
podman run -it --rm --name orchestrator \
  --network my-ai-network \
  -v /run/user/$(id -u)/pipewire-0:/run/user/1000/pipewire-0:z \
  -e PIPEWIRE_RUNTIME_DIR=/run/user/1000 \
  --device /dev/snd:/dev/snd \
  --group-add keep-groups \
  --user $(id -u):$(id -g) \
  localhost/orchestrator-image
