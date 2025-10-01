#!/bin/bash

# ==============================================================================
# Unified Management Script for the AI Assistant Project
#
# USAGE:
#   ./manage.sh run      - Builds and runs the main application stack.
#   ./manage.sh commit   - Uses a dedicated, separate LLM to generate a commit message.
#   ./manage.sh logs     - Tails the logs of all running services.
#   ./manage.sh stop     - Stops and removes all project containers.
#   ./manage.sh clean    - Stops containers and removes the network.
#
# ==============================================================================

# --- CONFIGURATION
NETWORK_NAME="my-ai-network"
RM_NAME="resource-manager"         && RM_PORT="8000" && RM_INTERNAL_PORT="8080"
ASR_NAME="asr-service"             && ASR_PORT="8001" && ASR_INTERNAL_PORT="9000"
TTS_NAME="tts-service"             && TTS_PORT="8002" && TTS_INTERNAL_PORT="8000"
EMBEDDING_NAME="embedding-service" && EMBEDDING_PORT="8003" && EMBEDDING_INTERNAL_PORT="8000"
RAG_NAME="rag-service"             && RAG_PORT="8004" && RAG_INTERNAL_PORT="8000"
LLM_NAME="llm-service"             && LLM_PORT="8080" && LLM_INTERNAL_PORT="8080"
NEO4J_NAME="neo4j-db"
ORCHESTRATOR_NAME="orchestrator-service"

# --- Helper Service for Git Commits ---
COMMIT_HELPER_NAME="commit-helper-service"
COMMIT_LLM_NAME="commit-helper-llm"
COMMIT_LLM_PORT="8081"
COMMIT_LLM_IMAGE="my-ai/commit-helper-service"
ALL_CONTAINERS="$RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME $ORCHESTRATOR_NAME $COMMIT_LLM_NAME"

# --- Colors for logs ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m';
MAGENTA='\033[0;35m'; CYAN='\033[0;36m'; WHITE='\033[0;37m'; ORANGE='\033[0;93m'; NC='\033[0m'

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

dump_diagnostics() {
    echo -e "${RED}🔴 A failure occurred or dump was requested. Collecting diagnostics...${NC}"
    echo "=================================================="
    echo "============== SYSTEM & GPU STATE =============="
    echo "=================================================="
    dmesg | tail -n 50 && echo "" && podman stats --no-stream
    echo "=================================================="
    echo "============== CONTAINER LOGS ===================="
    echo "=================================================="
    for container in $RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME; do
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
    kill $(jobs -p) &>/dev/null
}

stop_services() {
    echo -e "${YELLOW}### STOPPING AND REMOVING ALL CONTAINERS ###${NC}"
    podman stop --time=10 $ALL_CONTAINERS &>/dev/null && \
    podman rm $ALL_CONTAINERS &>/dev/null || true
    echo "✅ Containers removed."
}

tail_logs() {
    echo -e "${GREEN}### TAILING LOGS (Press Ctrl+C to exit) ###${NC}"
    tail_log_helper() {
        local name="$1"; local color="$2";
        local padded_name=$(printf "%-20s" "$name")
        podman logs -f "$name" 2>&1 | awk -v name="$padded_name" -v color="$color" -v nc="$NC" '{ print color "[" name "] " nc $0 }' &
    }
    tail_log_helper "$RM_NAME" "$GREEN"; tail_log_helper "$ASR_NAME" "$BLUE"; tail_log_helper "$TTS_NAME" "$MAGENTA";
    tail_log_helper "$EMBEDDING_NAME" "$CYAN"; tail_log_helper "$RAG_NAME" "$YELLOW"; tail_log_helper "$LLM_NAME" "$RED";
    tail_log_helper "$NEO4J_NAME" "$WHITE";
}

start_commit_llm() {
    if ! podman image exists "$COMMIT_LLM_IMAGE"; then
        echo -e "${CYAN}### Building dedicated commit helper image... ###${NC}"
        podman build -t $COMMIT_LLM_IMAGE ./$COMMIT_HELPER_NAME
    fi
    if ! podman container exists "$COMMIT_LLM_NAME" || [ "$(podman inspect -f '{{.State.Status}}' $COMMIT_LLM_NAME)" != "running" ]; then
        echo -e "${CYAN}### Starting dedicated commit helper LLM... ###${NC}"
        podman run --replace -d --name $COMMIT_LLM_NAME --network $NETWORK_NAME --gpus all -p $COMMIT_LLM_PORT:$LLM_INTERNAL_PORT \
          -v ./volumes/llm/gguf-models:/models:z \
          $COMMIT_LLM_IMAGE
        echo "Waiting for helper LLM..."
        until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$COMMIT_LLM_PORT/health)" = "200" ]; do sleep 1; done
        echo "✅ Commit helper is ready."
    else
        echo "✅ Commit helper is already running."
    fi
}

auto_commit() {
    echo -e "${GREEN}### CHECKING FOR CHANGES ###${NC}"
    if git diff-index --quiet HEAD --; then echo "✅ No changes detected."; return; fi
    git status -s; echo ""
    read -p "Proceed with auto-commit? (y/N) " -n 1 -r; echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then echo "Commit aborted."; return; fi
    start_commit_llm
    git add .
    DIFF_CONTENT=$(git diff --staged)
    PROMPT="You are an expert programmer writing conventional commit messages. Summarize the following git diff into a commit message. Format: <type>: <subject>\n\n<body>\n\nDiff:\n---\n$DIFF_CONTENT\n---"
    COMMIT_MSG=$(curl -s -X POST "http://localhost:$COMMIT_LLM_PORT/completion" -H 'Content-Type: application/json' -d "{ \"prompt\": \"$PROMPT\" }" | jq -r '.content')
    if [ -z "$COMMIT_MSG" ]; then COMMIT_MSG="feat: Auto-commit on $(date)"; fi
    echo -e "${GREEN}Generated Commit Message:${NC}\n${WHITE}$COMMIT_MSG${NC}\n"
    git commit -m "$COMMIT_MSG"
    read -p "Push to origin? (y/N) " -n 1 -r; echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then git push; fi
}

run_services() {
    set -e
    trap dump_diagnostics ERR
    
    echo -e "${GREEN}### PREPARING NETWORK ###${NC}"
    podman network create $NETWORK_NAME &>/dev/null || true

    echo -e "${GREEN}### BUILDING MAIN APPLICATION IMAGES ###${NC}"
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/resource-manager-service ./resource-manager-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/asr-service ./asr-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/llm-service ./llm-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/tts-service ./tts-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/embedding-service ./embedding-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/rag-service ./rag-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/orchestrator-service ./orchestrator-service
    echo "✅ Images built."

    mkdir -p ./volumes/asr/.cache ./volumes/tts/voices ./volumes/tts/model-cache \
           ./volumes/llm/gguf-models ./volumes/embedding/.cache ./volumes/neo4j/data \
           ./volumes/rag/input_data ./volumes/rag/graph_data

    stop_services

    echo -e "${GREEN}### STARTING SERVICES ###${NC}"
    podman run --replace -d --name $RM_NAME --network $NETWORK_NAME -p $RM_PORT:$RM_INTERNAL_PORT -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" -e TTS_SERVICE_URL="http://$TTS_NAME:$TTS_INTERNAL_PORT" -e EMBEDDING_SERVICE_URL="http://$EMBEDDING_NAME:$EMBEDDING_INTERNAL_PORT" my-ai/resource-manager-service
    podman run --replace -d --name $ASR_NAME --network $NETWORK_NAME -p $ASR_PORT:$ASR_INTERNAL_PORT --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/asr/.cache:/root/.cache:z my-ai/asr-service
    podman run --replace -d --name $NEO4J_NAME --network $NETWORK_NAME -v ./volumes/neo4j/data:/data:z -e NEO4J_AUTH=none -e NEO4J_PLUGINS='["apoc"]' docker.io/library/neo4j:latest
    podman run --replace -d --name $RAG_NAME --network $NETWORK_NAME -p $RAG_PORT:$RAG_INTERNAL_PORT --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/rag/input_data:/app/input_data:z -e NEO4J_URI="bolt://$NEO4J_NAME:7687" -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" -e EMBEDDING_SERVICE_URL="http://$EMBEDDING_NAME:$EMBEDDING_INTERNAL_PORT" -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" my-ai/rag-service
    podman run --replace -d --name $LLM_NAME --network $NETWORK_NAME --gpus all -p $LLM_PORT:$LLM_INTERNAL_PORT -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/llm/gguf-models:/models:z -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT"  my-ai/llm-service
    podman run --replace -d --name $TTS_NAME --network $NETWORK_NAME --gpus all -p $TTS_PORT:$TTS_INTERNAL_PORT -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/tts/voices:/voices:z -v ./volumes/tts/model-cache:/root/.local/share/tts:z -e COQUI_TOS_AGREED=1 -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" my-ai/tts-service
    podman run --replace -d --name $EMBEDDING_NAME --network $NETWORK_NAME --gpus all -p $EMBEDDING_PORT:$EMBEDDING_INTERNAL_PORT -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/embedding/.cache:/root/.cache:z -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" my-ai/embedding-service
    echo "✅ All background services started."
    
    sleep 5
    
    echo -e "\n${ORANGE}Launching Orchestrator in the foreground...${NC}"
    podman run -it --rm --name $ORCHESTRATOR_NAME --network $NETWORK_NAME --device /dev/snd \
      -e PIPEWIRE_RUNTIME_DIR="/run/user/$(id -u)" \
      -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" \
      -e ASR_SERVICE_URL="http://$ASR_NAME:$ASR_INTERNAL_PORT" \
      -e TTS_SERVICE_URL="http://$TTS_NAME:$TTS_INTERNAL_PORT" \
      -e RAG_SERVICE_URL="http://$RAG_NAME:$RAG_INTERNAL_PORT" \
      -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
      my-ai/orchestrator-service
}

# ==============================================================================
# COMMAND DISPATCHER
# ==============================================================================
trap cleanup EXIT

usage() {
    echo -e "${GREEN}USAGE:${NC}"
    grep -E "^#   \./manage.sh" "$0" | sed 's/^#   //'
}

case "$1" in
    run)
        run_services
        ;;
    commit)
        auto_commit
        ;;
    setup-commit-helper)
        setup_commit_helper
        ;;
    logs)
        tail_logs
        wait
        ;;
    stop)
        stop_services
        ;;
    clean)
        stop_services
        echo -e "${YELLOW}### REMOVING NETWORK ###${NC}"
        podman network rm $NETWORK_NAME &>/dev/null || true
        echo "✅ Network removed."
        ;;
    *)
        usage
        exit 1
        ;;
esac
