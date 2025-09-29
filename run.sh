#!/bin/bash

# This function will be called whenever a command exits with a non-zero status.
handle_failure() {
    echo "🔴 run.sh failed. Running dump.sh to collect diagnostics..."
    ./dump.sh
}

# This function will be called on script exit to clean up background processes.
cleanup() {
    echo "Killing background log tails..."
    # This will kill all background jobs started by this script.
    kill $(jobs -p) &>/dev/null
}


# Register the handle_failure function to be executed on ERR signal
trap handle_failure ERR
# Register the cleanup function to be executed on script exit.
trap cleanup EXIT


# Exit immediately if a command exits with a non-zero status.
set -e

# --- CONFIGURATION: SINGLE SOURCE OF TRUTH ---
NETWORK_NAME="my-ai-network"
RM_NAME="resource-manager"         && RM_PORT="8000" && RM_INTERNAL_PORT="8080"
ASR_NAME="asr-service"             && ASR_PORT="8001" && ASR_INTERNAL_PORT="9000"
TTS_NAME="tts-service"             && TTS_PORT="8002" && TTS_INTERNAL_PORT="8000"
EMBEDDING_NAME="embedding-service" && EMBEDDING_PORT="8003" && EMBEDDING_INTERNAL_PORT="8000"
RAG_NAME="rag-service"             && RAG_PORT="8004" && RAG_INTERNAL_PORT="8000"
LLM_NAME="llm-service"             && LLM_PORT="8080" && LLM_INTERNAL_PORT="8080"
NEO4J_NAME="neo4j-db"

# --- 0. PREPARE NETWORK ---
echo "##################################################"
echo "### PREPARING NETWORK"
echo "##################################################"
echo "Creating podman network '$NETWORK_NAME'..."
podman network create $NETWORK_NAME &>/dev/null || true
echo "✅ Network ready."
echo

# --- 1. BUILD IMAGES ---
echo "##################################################"
echo "### BUILDING CONTAINER IMAGES"
echo "##################################################"
podman build --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/resource-manager-service ./resource-manager-service
podman build --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/asr-service ./asr-service
podman build --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/llm-service ./llm-service
podman build --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/tts-service ./tts-service
podman build --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/embedding-service ./embedding-service
podman build --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/rag-service ./rag-service
podman build --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/orchestrator-service ./orchestrator-service
echo "✅ Images built."
echo

# --- 2. PREPARE DIRECTORIES ---
mkdir -p ./volumes/asr/.cache ./volumes/tts/voices ./volumes/tts/model-cache \
           ./volumes/llm/gguf-models ./volumes/embedding/.cache ./volumes/neo4j/data \
           ./volumes/rag/input_data ./volumes/rag/graph_data

# --- 3. STOP AND REMOVE OLD CONTAINERS ---
echo "##################################################"
echo "### CLEANING UP OLD CONTAINERS"
echo "##################################################"
podman stop --time=30 $RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME orchestrator-service &>/dev/null && \
podman rm  $RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME orchestrator-service &>/dev/null || true
echo "✅ Old containers removed."
echo

# --- 4. RUN SERVICES ---
echo "##################################################"
echo "### STARTING SERVICES"
echo "##################################################"

# Central Resource Manager
podman run --replace -d --name $RM_NAME --network $NETWORK_NAME -p $RM_PORT:$RM_INTERNAL_PORT \
  -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
  -e TTS_SERVICE_URL="http://$TTS_NAME:$TTS_INTERNAL_PORT" \
  -e EMBEDDING_SERVICE_URL="http://$EMBEDDING_NAME:$EMBEDDING_INTERNAL_PORT" \
  my-ai/resource-manager-service

# ASR Service (GPU)
podman run --replace -d --name $ASR_NAME --network $NETWORK_NAME -p $ASR_PORT:$ASR_INTERNAL_PORT --gpus all \
  -v ./volumes/asr/.cache:/root/.cache:z \
  my-ai/asr-service

# Neo4j Database (CPU)
podman run --replace -d --name $NEO4J_NAME --network $NETWORK_NAME \
  -v ./volumes/neo4j/data:/data:z -e NEO4J_AUTH=none \
  -e NEO4J_PLUGINS='["apoc"]' \
  docker.io/library/neo4j:latest

# RAG Service (GPU)
podman run --replace -d --name $RAG_NAME --network $NETWORK_NAME -p $RAG_PORT:$RAG_INTERNAL_PORT --gpus all \
  -v ./volumes/rag/input_data:/app/input_data:z \
  -v ./volumes/llm/gguf-models:/models:z \
  -v ./volumes/embedding/.cache:/root/.cache:z \
  -e NEO4J_URI="bolt://$NEO4J_NAME:7687" \
  -e LLM_MODEL_PATH="/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf" \
  -e EMBEDDING_MODEL_NAME="nomic-ai/nomic-embed-text-v1.5" \
  -e HF_HOME="/root/.cache/huggingface" \
  my-ai/rag-service

# LLM Service (GPU)
podman run --replace -d --name $LLM_NAME --network $NETWORK_NAME --gpus all -p $LLM_PORT:$LLM_INTERNAL_PORT \
  -v ./volumes/llm/gguf-models:/models:z \
  -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT"  \
  my-ai/llm-service

# TTS Service (GPU)
podman run --replace -d --name $TTS_NAME --network $NETWORK_NAME --gpus all -p $TTS_PORT:$TTS_INTERNAL_PORT \
  -v ./volumes/tts/voices:/voices:z \
  -v ./volumes/tts/model-cache:/root/.local/share/tts:z \
  -e COQUI_TOS_AGREED=1 \
  -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT"  \
  my-ai/tts-service

# Embedding Service (GPU)
podman run --replace -d --name $EMBEDDING_NAME --network $NETWORK_NAME --gpus all -p $EMBEDDING_PORT:$EMBEDDING_INTERNAL_PORT \
  -v ./volumes/embedding/.cache:/root/.cache:z \
  -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT"  \
  my-ai/embedding-service

echo "✅ All services started."
echo
echo "##################################################"
echo "### TAILING LOGS (Press Ctrl+C to exit)"
echo "##################################################"

# Wait a moment for all containers to initialize before tailing logs
sleep 5

# --- Function to tail logs with a colored prefix ---
tail_log() {
    local name="$1"
    local color="$2"
    # Pad the name to a consistent width for alignment
    local padded_name=$(printf "%-20s" "$name")
    podman logs -f "$name" 2>&1 | awk -v name="$padded_name" -v color="$color" '{
        print color "[" name "] \033[0m" $0
    }' &
}

# --- Colors for logs ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
ORANGE='\033[0;93m'


# --- Tail logs in the background with prefixes and colors ---
tail_log "$RM_NAME" "$GREEN"
tail_log "$ASR_NAME" "$BLUE"
tail_log "$TTS_NAME" "$MAGENTA"
tail_log "$EMBEDDING_NAME" "$CYAN"
# For RAG, filter out the noisy llama_model_loader and control token messages
podman logs -f "$RAG_NAME" 2>&1 | grep -Ev "llama_model_loader|control token" | awk -v name="$(printf '%-20s' "$RAG_NAME")" -v color="$YELLOW" '{
    print color "[" name "] \033[0m" $0
}' &
tail_log "$LLM_NAME" "$RED"
tail_log "$NEO4J_NAME" "$WHITE"


# --- Orchestrator Service (runs in foreground) ---
echo -e "\n${ORANGE}Launching Orchestrator in the foreground...${WHITE}\n"

podman run -it --rm --name orchestrator-service --network $NETWORK_NAME --device /dev/snd \
  --user $(id -u):$(id -g) \
  --group-add audio \
  --group-add video \
  -v /run/user/$(id -u)/pipewire-0:/run/user/$(id -u)/pipewire-0 \
  -e PIPEWIRE_RUNTIME_DIR="/run/user/$(id -u)" \
  -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
  -e ASR_SERVICE_URL="http://$ASR_NAME:$ASR_INTERNAL_PORT" \
  -e TTS_SERVICE_URL="http://$TTS_NAME:$TTS_INTERNAL_PORT" \
  -e RAG_SERVICE_URL="http://$RAG_NAME:$RAG_INTERNAL_PORT" \
  -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
  my-ai/orchestrator-service
