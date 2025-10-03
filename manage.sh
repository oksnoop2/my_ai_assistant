#!/bin/bash

# ==============================================================================
# Unified Management Script for the AI Assistant Project ~
#
# USAGE:
#   ./manage.sh run      - Builds and runs the main application stack with live logs.
#   ./manage.sh commit   - Uses a dedicated, separate LLM to generate a commit message.
#   ./manage.sh logs     - Tails the logs of all running services.
#   ./manage.sh stop     - Stops and removes all project containers.
#   ./manage.sh clean    - Stops containers and removes the network.
#   ./manage.sh dump     - Dumps diagnostic information for debugging.
#
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
LLM_NAME="llm-service"             && LLM_PORT="8080" && LLM_INTERNAL_PORT="8080"
NEO4J_NAME="neo4j-db"
ORCHESTRATOR_NAME="orchestrator-service"

# --- Helper Service for Git Commits ---
COMMIT_HELPER_NAME="commit-helper-service"
COMMIT_LLM_NAME="commit-helper-llm"
COMMIT_LLM_PORT="8081"
COMMIT_LLM_INTERNAL_PORT="8080"
COMMIT_LLM_IMAGE="my-ai/commit-helper-service"

ALL_CONTAINERS="$RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME $ORCHESTRATOR_NAME $COMMIT_LLM_NAME $EMOTION_CLASSIFIER_NAME $PERSONA_NAME"

# --- Colors for logs ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m';
MAGENTA='\033[0;35m'; CYAN='\033[0;36m'; WHITE='\033[0;37m'; ORANGE='\033[0;93m'; PURPLE='\033[0;35m'; NC='\033[0m'

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

usage() {
    echo "USAGE:"
    echo "  $0 run      - Builds and runs the main application stack with live logs."
    echo "  $0 commit   - Uses a dedicated, separate LLM to generate a commit message."
    echo "  $0 logs     - Tails the logs of all running services."
    echo "  $0 stop     - Stops and removes all project containers."
    echo "  $0 clean    - Stops containers and removes the network."
    echo "  $0 dump     - Dumps diagnostic information for debugging."
}

dump_diagnostics() {
    echo -e "${RED}🔴 A failure occurred or dump was requested. Collecting diagnostics...${NC}"
    echo "=================================================="
    echo "============== SYSTEM & GPU STATE =============="
    echo "=================================================="
    dmesg | tail -n 50 && echo "" && podman stats --no-stream
    echo "=================================================="
    echo "============== CONTAINER LOGS ===================="
    echo "=================================================="
    for container in $RM_NAME $ASR_NAME $TTS_NAME $EMBEDDING_NAME $RAG_NAME $LLM_NAME $NEO4J_NAME $EMOTION_CLASSIFIER_NAME $PERSONA_NAME; do
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
    tail_log_helper "$NEO4J_NAME" "$WHITE"; tail_log_helper "$EMOTION_CLASSIFIER_NAME" "$ORANGE"; tail_log_helper "$PERSONA_NAME" "$PURPLE";
}

# --- RESTORED: Commit helper functions ---
start_commit_llm() {
    local build_image=false
    if ! podman image exists "$COMMIT_LLM_IMAGE"; then
        build_image=true
    else
        local image_created_time
        image_created_time=$(podman inspect -f '{{.Created.Format "2006-01-02 15:04:05"}}' "$COMMIT_LLM_IMAGE")
        
        if [ "$(find "./$COMMIT_HELPER_NAME" -type f -newermt "$image_created_time" | wc -l)" -gt 0 ]; then
            echo -e "${YELLOW}### Source files have changed. Rebuilding commit helper image... ###${NC}"
            build_image=true
        fi
    fi

    if [ "$build_image" = true ]; then
        echo -e "${CYAN}### Building dedicated commit helper image... ###${NC}"
        podman build -t "$COMMIT_LLM_IMAGE" "./$COMMIT_HELPER_NAME"
    fi

    if ! podman container exists "$COMMIT_LLM_NAME" || [ "$(podman inspect -f '{{.State.Status}}' "$COMMIT_LLM_NAME")" != "running" ]; then
        echo -e "${CYAN}### Starting dedicated commit helper LLM... ###${NC}"
        podman run --replace -d --name "$COMMIT_LLM_NAME" --network "$NETWORK_NAME" --gpus all \
          -p "$COMMIT_LLM_PORT:$COMMIT_LLM_INTERNAL_PORT" \
          -v "./volumes/llm/gguf-models:/models:z" \
          "$COMMIT_LLM_IMAGE"
        
        echo "Waiting for helper LLM server..."
        until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$COMMIT_LLM_PORT/health")" = "200" ]; do 
            sleep 1; 
        done
        echo "✅ LLM Server is running."

        echo "⏳ Triggering model load in helper LLM..."
        sleep 1 
        curl -s --max-time 180 -X POST "http://localhost:$COMMIT_LLM_PORT/load" > /dev/null

        echo "Waiting for model to load..."
        until curl -s "http://localhost:$COMMIT_LLM_PORT/health" | jq -e '.model_loaded == true' > /dev/null; do
            printf "."
            sleep 1;
        done
        echo -e "\n✅ Commit helper model is loaded and ready."
    else
        echo "✅ Commit helper is already running."
    fi
}

auto_commit() {
    trap 'echo -e "\n${ORANGE}### Stopping commit helper service... ###${NC}"; podman stop "$COMMIT_LLM_NAME" &>/dev/null' RETURN

    echo -e "${GREEN}### CHECKING FOR CHANGES ###${NC}"
    if [[ -z $(git status --porcelain) ]]; then
        echo "✅ No changes detected."
        return
    fi
    git status -s
    echo ""
    
    read -p "Proceed with auto-commit? (y/N) " -r choice
    if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        echo "Commit aborted."
        return
    fi
    
    start_commit_llm
    git add .
    
    local DIFF_CONTENT
    DIFF_CONTENT=$(git diff --staged | grep -E '^\+|-|^\@\@' | grep -v -E '^\+\+\+|^\-\-\-')
    
    if [[ -z "$DIFF_CONTENT" ]]; then
        echo "No meaningful code changes detected to commit."
        return
    fi

    local PROMPT
    read -r -d '' PROMPT << EOM
You are an expert programmer writing a Conventional Commit message. Your task is to summarize the code changes below.
Follow these rules:
1. The message MUST follow the conventional commit format: \`<type\>(\<scope\>): \<subject\>\`.
2. Use the function name from the diff hunk (the text after '@@ ... @@') as the <scope>. If multiple functions are changed, use the most significant one.
3. The <subject> should be a concise, imperative summary of the change.
4. The body should explain the 'what' and 'why' of the changes. Do not mention filenames.
---
Code Changes:
---
$DIFF_CONTENT
---
Commit Message:
EOM

    local JSON_PAYLOAD
    JSON_PAYLOAD=$(jq -n --arg prompt_text "$PROMPT" '{prompt: $prompt_text}')
    
    echo "### GENERATING COMMIT MESSAGE (this may take a moment)... ###"
    local COMMIT_MSG
    COMMIT_MSG=$(curl -s --max-time 120 -X POST "http://localhost:$COMMIT_LLM_PORT/completion" \
        -H "Content-Type: application/json" \
        -d "$JSON_PAYLOAD" | jq -r '.content')
        
    COMMIT_MSG="${COMMIT_MSG#"${COMMIT_MSG%%[![:space:]]*}"}"
    COMMIT_MSG="${COMMIT_MSG%"${COMMIT_MSG##*[![:space:]]}"}"
        
    if [[ -z "$COMMIT_MSG" ]]; then
        echo -e "${RED}🔥 Failed to generate commit message. Using a default message.${NC}"
        COMMIT_MSG="chore: automatic commit on $(date)"
    fi
    
    echo -e "${GREEN}Generated Commit Message:${NC}\n${WHITE}$COMMIT_MSG${NC}\n"
    
    git commit -m "$COMMIT_MSG"
    
    echo -e "\n${GREEN}✅ Commit created locally. Push to remote using your Git client.${NC}"
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
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/emotion-classifier-service ./emotion-classifier-service
    podman build -q --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/persona-service ./persona-service
    podman build -q --no-cache --volume $PWD/volumes/pip-cache:/cache:z -t my-ai/orchestrator-service ./orchestrator-service
    echo "✅ Images built."

    mkdir -p ./volumes/asr/.cache ./volumes/tts/voices ./volumes/tts/model-cache \
           ./volumes/llm/gguf-models ./volumes/embedding/.cache ./volumes/neo4j/data \
           ./volumes/rag/input_data ./volumes/rag/graph_data ./volumes/emotion-classifier/.cache \
           ./volumes/persona-service

    stop_services

    echo -e "${GREEN}### STARTING SERVICES ###${NC}"
    podman run --replace -d --name $RM_NAME --network $NETWORK_NAME -p $RM_PORT:$RM_INTERNAL_PORT -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" -e TTS_SERVICE_URL="http://$TTS_NAME:$TTS_INTERNAL_PORT" -e EMBEDDING_SERVICE_URL="http://$EMBEDDING_NAME:$EMBEDDING_INTERNAL_PORT" my-ai/resource-manager-service
    podman run --replace -d --name $ASR_NAME --network $NETWORK_NAME -p $ASR_PORT:$ASR_INTERNAL_PORT --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/asr/.cache:/root/.cache:z my-ai/asr-service
    podman run --replace -d --name $NEO4J_NAME --network $NETWORK_NAME -v ./volumes/neo4j/data:/data:z -e NEO4J_AUTH=none -e NEO4J_PLUGINS='["apoc"]' docker.io/library/neo4j:latest
    podman run --replace -d --name $RAG_NAME --network $NETWORK_NAME -p $RAG_PORT:$RAG_INTERNAL_PORT --gpus all -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/rag/input_data:/app/input_data:z -e NEO4J_URI="bolt://$NEO4J_NAME:7687" -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" -e EMBEDDING_SERVICE_URL="http://$EMBEDDING_NAME:$EMBEDDING_INTERNAL_PORT" -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" my-ai/rag-service
    podman run --replace -d --name $LLM_NAME --network $NETWORK_NAME --gpus all -p $LLM_PORT:$LLM_INTERNAL_PORT -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/llm/gguf-models:/models:z -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT"  my-ai/llm-service
    podman run --replace -d --name $TTS_NAME --network $NETWORK_NAME --gpus all -p $TTS_PORT:$TTS_INTERNAL_PORT -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/tts/voices:/voices:z -v ./volumes/tts/model-cache:/root/.local/share/tts:z -e COQUI_TOS_AGREED=1 -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" my-ai/tts-service
    podman run --replace -d --name $EMBEDDING_NAME --network $NETWORK_NAME --gpus all -p $EMBEDDING_PORT:$EMBEDDING_INTERNAL_PORT -e NVIDIA_VISIBLE_DEVICES=all -v ./volumes/embedding/.cache:/root/.cache:z -e RESOURCE_MANAGER_URL="http://$RM_NAME:$RM_INTERNAL_PORT" my-ai/embedding-service
    podman run --replace -d --name $EMOTION_CLASSIFIER_NAME --network $NETWORK_NAME -p $EMOTION_CLASSIFIER_PORT:$EMOTION_CLASSIFIER_INTERNAL_PORT -v ./volumes/emotion-classifier/.cache:/root/.cache:z my-ai/emotion-classifier-service
    podman run --replace -d --name $PERSONA_NAME --network $NETWORK_NAME -p $PERSONA_PORT:$PERSONA_INTERNAL_PORT \
      -e LLM_SERVICE_URL="http://$LLM_NAME:$LLM_INTERNAL_PORT" \
      -e EMOTION_CLASSIFIER_URL="http://$EMOTION_CLASSIFIER_NAME:$EMOTION_CLASSIFIER_INTERNAL_PORT" \
      -e RAG_SERVICE_URL="http://$RAG_NAME:$RAG_INTERNAL_PORT" \
      my-ai/persona-service
      
    echo "✅ All background services started."
    
    # --- NEW: Launch the real-time log viewer ---
    tail_logs
    
    echo -e "\n${ORANGE}Launching Orchestrator in the foreground...${NC}"
    podman run -it --rm --name $ORCHESTRATOR_NAME --network $NETWORK_NAME \
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
}

# ==============================================================================
# COMMAND DISPATCHER
# ==============================================================================
trap cleanup EXIT

case "$1" in
    run)
        run_services
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
