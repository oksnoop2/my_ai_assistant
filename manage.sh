#!/bin/bash

# ==============================================================================
# Unified Management Script for the AI Assistant Project ~
# ==============================================================================

# --- CONFIGURATION
NETWORK_NAME="my-ai-network"
RM_NAME="resource-manager"         && RM_PORT="8000" && RM_INTERNAL_PORT="8080"
ASR_NAME="asr-service"             && ASR_PORT="8001" && ASR_INTERNAL_PORT="9000"
TTS_NAME="tts-service"             && TTS_PORT="8002" && TTS_INTERNAL_PORT="8000"
EMBEDDING_NAME="embedding-service" && EMBEDDING_PORT="8003" && EMBEDDING_INTERNAL_PORT="8000"
RAG_NAME="rag-service"             && RAG_PORT="8004" && RAG_INTERNAL_PORT="8000"
EMOTION_CLASSIFIER_NAME="emotion-classifier-service" && EMOTION_CLASSIFIER_PORT="8005" && EMOTION_CLASSIFIER_INTERNAL_PORT="8000"
PERSONA_NAME="persona-service"     && PERSONA_PORT="8006" && PERSONA_INTERNAL_PORT="8000"
INGESTION_NAME="ingestion-service" && INGESTION_PORT="8007" && INGESTION_INTERNAL_PORT="8007"
LLM_NAME="llm-service"             && LLM_PORT="8080" && LLM_INTERNAL_PORT="8080"
NEO4J_NAME="neo4j-db"
ORCHESTRATOR_NAME="orchestrator-service"

# --- Helper Service for Git Commits ---
COMMIT_HELPER_NAME="commit-helper-service"
COMMIT_LLM_NAME="commit-helper-llm"
COMMIT_LLM_PORT="8081"
COMMIT_LLM_INTERNAL_PORT="8080"
COMMIT_LLM_IMAGE="my-ai/commit-helper-service"

# --- List of all services to be tailed ---
BACKGROUND_SERVICES="$RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME $EMOTION_CLASSIFIER_NAME $PERSONA_NAME $INGESTION_NAME"
ALL_SERVICES_TO_LOG="$BACKGROUND_SERVICES $ORCHESTRATOR_NAME"
ALL_CONTAINERS="$BACKGROUND_SERVICES $ORCHESTRATOR_NAME $COMMIT_LLM_NAME"

# --- Colors for logs ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m';
MAGENTA='\033[0;35m'; CYAN='\033[0;36m'; WHITE='\033[0;37m'; ORANGE='\033[0;93m'; PURPLE='\033[0;35m'; NC='\033[0m'

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================
usage() {
    echo "USAGE:"
    echo "  $0 run      - Starts all services in a split-pane tmux session."
    echo "  $0 start    - Builds and starts all services in the background without logging."
    echo "  $0 attach   - Attaches to the interactive orchestrator if it's already running."
    echo "  $0 logs     - Tails the logs of all currently running services."
    echo "  $0 commit   - Uses a dedicated, separate LLM to generate a commit message."
    echo "  $0 stop     - Stops and removes all project containers and the tmux session."
    echo "  $0 clean    - Stops containers/session and removes the network."
    echo "  $0 dump     - Dumps diagnostic information for debugging."
}

dump_diagnostics() {
    echo -e "${RED}🔴 A failure occurred or dump was requested. Collecting diagnostics...${NC}"
    echo "=================================================="
    echo "============== SYSTEM & GPU STATE =============="
    echo "=================================================="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi || echo "nvidia-smi failed"
    else
        echo "nvidia-smi not found"
    fi
    echo ""
    dmesg | tail -n 50 && echo "" && podman stats --no-stream
    echo "=================================================="
    echo "============== CONTAINER LOGS ===================="
    echo "=================================================="
    for container in $RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME $EMOTION_CLASSIFIER_NAME $PERSONA_NAME $INGESTION_NAME; do
        if podman container exists "$container"; then
            echo -e "\n${YELLOW}### LOGS FOR CONTAINER: $container ###${NC}"
            podman logs --tail 200 "$container" || echo "--> Failed to retrieve logs for $container."
        fi
    done
    echo "=================================================="
    echo "========== SOURCE CODE & DOCKERFILES ==========="
    echo "=================================================="
    find . -path './.git' -prune -o -path '*/__pycache__' -prune -o -path './volumes' -prune -o \( -name "*.py" -o -name "Dockerfile" \) -print | sort | while read filename; do
        echo -e "\n${CYAN}######################################################################${NC}"
        echo -e "${CYAN}### FILE: ${YELLOW}$filename${NC}"
        echo -e "${CYAN}######################################################################${NC}"
        cat -n "$filename"
    done
    echo "=================================================="
    echo "Analysis complete."
    echo "=================================================="
}

cleanup() {
    echo -e "\n${ORANGE}Script finished. Cleaning up background processes...${NC}"
    if ((${#PIDS[@]})); then
      kill "${PIDS[@]}" &>/dev/null
    fi
}

stop_services() {
    local kill_tmux=${1:-false}
    
    echo -e "${YELLOW}### STOPPING AND REMOVING ALL CONTAINERS ###${NC}"
    podman stop --time=10 $ALL_CONTAINERS &>/dev/null && \
    podman rm $ALL_CONTAINERS &>/dev/null || true
    echo "✅ Containers removed."

    if [ "$kill_tmux" = true ] && command -v tmux &> /dev/null && tmux has-session &> /dev/null; then
        echo "Shutting down tmux server..."
        tmux kill-server &> /dev/null
    fi
}

tail_logs() {
    PIDS=()
    echo "### Starting log streams... ###"
    sleep 2

    declare -A COLORS=(
        ["$RM_NAME"]="$YELLOW" ["$ASR_NAME"]="$CYAN" ["$TTS_NAME"]="$MAGENTA"
        ["$RAG_NAME"]="$GREEN" ["$LLM_NAME"]="$RED" ["$PERSONA_NAME"]="$PURPLE"
        ["$EMBEDDING_NAME"]="$GREEN" ["$EMOTION_CLASSIFIER_NAME"]="$MAGENTA"
        ["$NEO4J_NAME"]="$BLUE" ["$ORCHESTRATOR_NAME"]="$WHITE" ["$INGESTION_NAME"]="$ORANGE"
    )

    for name in $ALL_SERVICES_TO_LOG; do
        if podman container exists "$name"; then
            local color=${COLORS[$name]:-$WHITE}
            {
                stdbuf -oL podman logs -f "$name" | while IFS= read -r line; do
                    printf "\r\e[K${color}%-28s${NC} │ %s\n" "$name" "$(echo -e "$line" | tr -d '\r')"
                done
            } &
            PIDS+=($!)
        else
            echo -e "${RED}Warning: Container '$name' not found for log tailing.${NC}"
        fi
    done
}

check_gpu_support() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}⚠️  nvidia-smi not found. GPU support may not be available.${NC}"
        echo "If you have an NVIDIA GPU, please install the NVIDIA drivers and container toolkit."
        return 1
    fi
    
    echo -e "${CYAN}Testing GPU access in container...${NC}"
    if podman run --rm --gpus all --security-opt=label=disable nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo -e "${GREEN}✅ GPU access confirmed in containers.${NC}"
        return 0
    else
        echo -e "${RED}❌ GPU access failed in containers. Checking container toolkit...${NC}"
        echo "Please ensure nvidia-container-toolkit is installed and configured for podman."
        echo "Run: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
        return 1
    fi
}

start_all_services() {
    echo -e "${GREEN}### CHECKING GPU SUPPORT ###${NC}"
    check_gpu_support
    GPU_AVAILABLE=$?

    echo -e "${GREEN}### PREPARING NETWORK ###${NC}"
    podman network create $NETWORK_NAME &>/dev/null || true

    echo -e "${GREEN}### BUILDING MAIN APPLICATION IMAGES ###${NC}"
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/resource-manager-service ./resource-manager-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/asr-service ./asr-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/llm-service ./llm-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/tts-service ./tts-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/embedding-service ./embedding-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/rag-service ./rag-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/emotion-classifier-service ./emotion-classifier-service
    podman build --no-cache -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/persona-service ./persona-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/ingestion-service ./ingestion-service
    podman build -q --no-cache --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/orchestrator-service ./orchestrator-service
    echo "✅ Images built."

    mkdir -p ./volumes/asr/.cache ./volumes/tts/voices ./volumes/tts/model-cache \
           ./volumes/llm/gguf-models ./volumes/embedding/.cache ./volumes/neo4j/data \
           ./volumes/rag/input_data ./volumes/rag/graph_data ./volumes/emotion-classifier/.cache \
           ./volumes/persona-service ./volumes/ingestion/input

    stop_services

    echo -e "${GREEN}### STARTING ALL SERVICES IN BACKGROUND ###${NC}"
    
    if [ $GPU_AVAILABLE -eq 0 ]; then
        GPU_FLAGS="--gpus all --security-opt=label=disable -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all"
        echo -e "${GREEN}Starting services with GPU support enabled.${NC}"
    else
        GPU_FLAGS=""
        echo -e "${YELLOW}Starting services without GPU support (CPU mode).${NC}"
    fi

    # Resource Manager (no GPU needed)
    podman run --replace -d --name $RM_NAME --network $NETWORK_NAME -p $RM_PORT:$RM_INTERNAL_PORT \
      -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
      -e TTS_SERVICE_URL="http://$TTS_NAME:$TTS_INTERNAL_PORT" \
      -e EMBEDDING_SERVICE_URL="http://$EMBEDDING_NAME:$EMBEDDING_INTERNAL_PORT" \
      my-ai/resource-manager-service
    
    # ASR Service (GPU-enabled)
    podman run --replace -d --name $ASR_NAME --network $NETWORK_NAME -p $ASR_PORT:$ASR_INTERNAL_PORT \
      $GPU_FLAGS \
      -v ./volumes/asr/.cache:/root/.cache:z \
      -v ./volumes/llm/gguf-models:/models:z \
      -v ./volumes/tts/voices:/voices:z \
      my-ai/asr-service
    
    # Neo4j (no GPU needed)
    podman run --replace -d --name $NEO4J_NAME --network $NETWORK_NAME \
      -v ./volumes/neo4j/data:/data:z \
      -e NEO4J_AUTH=none \
      -e NEO4J_PLUGINS='["apoc"]' \
      docker.io/library/neo4j:latest
    
    # RAG Service (GPU-enabled)
    podman run --replace -d --name $RAG_NAME --network $NETWORK_NAME -p $RAG_PORT:$RAG_INTERNAL_PORT \
      $GPU_FLAGS \
      -v ./volumes/rag/input_data:/app/input_data:z \
      -e NEO4J_URI="bolt://$NEO4J_NAME:7687" \
      -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
      -e EMBEDDING_SERVICE_URL="http://$EMBEDDING_NAME:$EMBEDDING_INTERNAL_PORT" \
      -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
      my-ai/rag-service
    
    # LLM Service (GPU-enabled)
    podman run --replace -d --name $LLM_NAME --network $NETWORK_NAME -p $LLM_PORT:$LLM_INTERNAL_PORT \
      $GPU_FLAGS \
      -v ./volumes/llm/gguf-models:/models:z \
      -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
      my-ai/llm-service
    
    # TTS Service (GPU-enabled)
    podman run --replace -d --name $TTS_NAME --network $NETWORK_NAME -p $TTS_PORT:$TTS_INTERNAL_PORT \
      $GPU_FLAGS \
      -v ./volumes/tts/voices:/voices:z \
      -v ./volumes/tts/model-cache:/root/.local/share/tts:z \
      -e COQUI_TOS_AGREED=1 \
      -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
      my-ai/tts-service
    
    # Embedding Service (GPU-enabled)
    podman run --replace -d --name $EMBEDDING_NAME --network $NETWORK_NAME -p $EMBEDDING_PORT:$EMBEDDING_INTERNAL_PORT \
      $GPU_FLAGS \
      -v ./volumes/embedding/.cache:/root/.cache:z \
      -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
      my-ai/embedding-service
    
    # Emotion Classifier (CPU only)
    podman run --replace -d --name $EMOTION_CLASSIFIER_NAME --network $NETWORK_NAME -p $EMOTION_CLASSIFIER_PORT:$EMOTION_CLASSIFIER_INTERNAL_PORT \
      -v ./volumes/emotion-classifier/.cache:/root/.cache:z \
      my-ai/emotion-classifier-service
    
    # Persona Service (no GPU needed)
    podman run --replace -d --name $PERSONA_NAME --network $NETWORK_NAME -p $PERSONA_PORT:$PERSONA_INTERNAL_PORT \
      -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
      -e EMOTION_CLASSIFIER_URL="http://$EMOTION_CLASSIFIER_NAME:$EMOTION_CLASSIFIER_INTERNAL_PORT" \
      -e RAG_SERVICE_URL="http://$RAG_NAME:$RAG_INTERNAL_PORT" \
      my-ai/persona-service
    
    # Ingestion Service (no GPU needed for processing, but calls LLM for vision)
    podman run --replace -d --name $INGESTION_NAME --network $NETWORK_NAME -p $INGESTION_PORT:$INGESTION_INTERNAL_PORT \
      -v ./volumes/ingestion/input:/app/input:z \
      -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
      -e RAG_SERVICE_URL="http://$RAG_NAME:$RAG_INTERNAL_PORT" \
      -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
      my-ai/ingestion-service
      
    echo "✅ All services started."
}

# --- Commit helper functions ---
start_commit_llm() {
    local build_image=false
    if ! podman image exists "$COMMIT_LLM_IMAGE"; then build_image=true; else
        local image_created_time=$(podman inspect -f '{{.Created.Format "2006-01-02 15:04:05"}}' "$COMMIT_LLM_IMAGE")
        if [ "$(find "./$COMMIT_HELPER_NAME" -type f -newermt "$image_created_time" | wc -l)" -gt 0 ]; then
            echo -e "${YELLOW}### Source files have changed. Rebuilding commit helper image... ###${NC}"; build_image=true;
        fi
    fi
    if [ "$build_image" = true ]; then
        echo -e "${CYAN}### Building dedicated commit helper image... ###${NC}"; podman build -t "$COMMIT_LLM_IMAGE" "./$COMMIT_HELPER_NAME";
    fi
    
    if [ $GPU_AVAILABLE -eq 0 ]; then
        GPU_FLAGS="--gpus all --security-opt=label=disable"
    else
        GPU_FLAGS=""
    fi
    
    if ! podman container exists "$COMMIT_LLM_NAME" || [ "$(podman inspect -f '{{.State.Status}}' "$COMMIT_LLM_NAME")" != "running" ]; then
        echo -e "${CYAN}### Starting dedicated commit helper LLM... ###${NC}"
        podman run --replace -d --name "$COMMIT_LLM_NAME" --network "$NETWORK_NAME" $GPU_FLAGS \
          -p "$COMMIT_LLM_PORT:$COMMIT_LLM_INTERNAL_PORT" -v "./volumes/llm/gguf-models:/models:z" "$COMMIT_LLM_IMAGE"
        echo "Waiting for helper LLM server..."; until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$COMMIT_LLM_PORT/health")" = "200" ]; do sleep 1; done; echo "✅ LLM Server is running."
        echo "⏳ Triggering model load in helper LLM..."; sleep 1; curl -s --max-time 180 -X POST "http://localhost:$COMMIT_LLM_PORT/load" > /dev/null
        echo "Waiting for model to load..."; until curl -s "http://localhost:$COMMIT_LLM_PORT/health" | jq -e '.model_loaded == true' > /dev/null; do printf "."; sleep 1; done; echo -e "\n✅ Commit helper model is loaded and ready."
    else echo "✅ Commit helper is already running."; fi
}

auto_commit() {
    trap 'echo -e "\n${ORANGE}### Stopping commit helper service... ###${NC}"; podman stop "$COMMIT_LLM_NAME" &>/dev/null' RETURN
    echo -e "${GREEN}### CHECKING FOR CHANGES ###${NC}"; if [[ -z $(git status --porcelain) ]]; then echo "✅ No changes detected."; return; fi
    git status -s; echo ""
    read -p "Proceed with auto-commit? (y/N) " -r choice; if [[ ! "$choice" =~ ^[Yy]$ ]]; then echo "Commit aborted."; return; fi
    start_commit_llm; git add .
    local DIFF_CONTENT=$(git diff --staged | grep -E '^\+|-|^\@\@' | grep -v -E '^\+\+\+|^\-\-\-')
    if [[ -z "$DIFF_CONTENT" ]]; then echo "No meaningful code changes detected to commit."; return; fi
    local PROMPT; read -r -d '' PROMPT << EOM
You are an expert programmer writing a Conventional Commit message. Your task is to summarize the code changes below.
Rules: The message MUST follow conventional commit format: \`<type\>(\<scope\>): \<subject\>\`. Use the function name from the diff as the <scope>. The <body> should explain the 'what' and 'why'. Do not mention filenames.
---
Code Changes:
---
$DIFF_CONTENT
---
Commit Message:
EOM
    local JSON_PAYLOAD=$(jq -n --arg prompt_text "$PROMPT" '{prompt: $prompt_text}')
    echo "### GENERATING COMMIT MESSAGE... ###"; local COMMIT_MSG=$(curl -s --max-time 120 -X POST "http://localhost:$COMMIT_LLM_PORT/completion" -H "Content-Type: application/json" -d "$JSON_PAYLOAD" | jq -r '.content')
    COMMIT_MSG="${COMMIT_MSG#"${COMMIT_MSG%%[![:space:]]*}"}"; COMMIT_MSG="${COMMIT_MSG%"${COMMIT_MSG##*[![:space:]]}"}"
    if [[ -z "$COMMIT_MSG" ]]; then echo -e "${RED}🔥 Failed to generate commit message. Using a default.${NC}"; COMMIT_MSG="chore: automatic commit on $(date)"; fi
    echo -e "${GREEN}Generated Commit Message:${NC}\n${WHITE}$COMMIT_MSG${NC}\n"; git commit -m "$COMMIT_MSG"
    echo -e "\n${GREEN}✅ Commit created locally. Push to remote using your Git client.${NC}"
}

# ==============================================================================
# COMMAND DISPATCHER
# ==============================================================================
trap cleanup EXIT

case "$1" in
    run)
        if ! command -v tmux &> /dev/null; then
            echo -e "${RED}Error: tmux is not installed. Please install it to use this feature.${NC}"
            exit 1
        fi

        if [ -z "$TMUX" ]; then
            if tmux has-session -t "my-ai-assistant" 2>/dev/null; then
                echo "Killing existing tmux session..."
                tmux kill-session -t "my-ai-assistant"
            fi
            echo "Relaunching in a new tmux session..."
            exec tmux new-session -s "my-ai-assistant" "./manage.sh run"
            exit 0
        fi

        trap "stop_services true" SIGINT ERR
        start_all_services
        
        echo "Setting up tmux panes..."
        tmux split-window -h
        
        tmux send-keys -t 1 "./manage.sh logs" C-m 
        
        tmux select-pane -t 0
        echo -e "\n${ORANGE}Starting orchestrator in interactive mode...${NC}"

        podman run --replace -it --name $ORCHESTRATOR_NAME --network $NETWORK_NAME \
          -v "/run/user/$(id -u)/pipewire-0:/run/user/$(id -u)/pipewire-0:ro" \
          -v "./volumes/tts/voices:/voices:z" \
          -e PIPEWIRE_RUNTIME_DIR="/run/user/$(id -u)" \
          -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
          -e ASR_SERVICE_URL="http://$ASR_NAME:$ASR_INTERNAL_PORT" \
          -e TTS_SERVICE_URL="http://$TTS_NAME:$TTS_INTERNAL_PORT" \
          -e RAG_SERVICE_URL="http://$RAG_NAME:$RAG_INTERNAL_PORT" \
          -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
          -e EMOTION_CLASSIFIER_URL="http://$EMOTION_CLASSIFIER_NAME:$EMOTION_CLASSIFIER_INTERNAL_PORT" \
          -e PERSONA_SERVICE_URL="http://$PERSONA_NAME:$PERSONA_INTERNAL_PORT" \
          my-ai/orchestrator-service
        ;;
    start)
        start_all_services
        echo -e "\n${GREEN}✅ All services are running in the background.${NC}"
        ;;
    attach)
        echo -e "\n${ORANGE}Attempting to attach to orchestrator... (Use Ctrl+P, Ctrl+Q to detach and leave running)${NC}"
        if ! podman container exists "$ORCHESTRATOR_NAME" || [ "$(podman inspect -f '{{.State.Status}}' "$ORCHESTRATOR_NAME")" != "running" ]; then
            echo -e "${RED}Error: Orchestrator container is not running. Use './manage.sh run' or './manage.sh start' first.${NC}"
            exit 1
        fi
        podman attach "$ORCHESTRATOR_NAME"
        ;;
    commit)
        auto_commit
        ;;
    dump)
        dump_diagnostics
        ;;
    logs)
        tail_logs
        wait
        ;;
    stop)
        stop_services true
        ;;
    clean)
        stop_services true
        echo -e "${YELLOW}### REMOVING NETWORK ###${NC}"
        podman network rm $NETWORK_NAME &>/dev/null || true
        echo "✅ Network removed."
        ;;
    *)
        usage
        exit 1
        ;;
esac
