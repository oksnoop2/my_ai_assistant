#!/bin/bash

# Enhanced Project Analysis Script
# Displays project analysis output directly in the terminal.

echo "Starting project analysis... Output will be displayed on the terminal."

# Function to output to the terminal
output() {
    echo "$1"
}

# --- NEW: DUMP CONTAINER LOGS ---
output "=================================================="
output "============== CONTAINER LOGS ===================="
output "=================================================="

# Get all container names associated with the project's network
CONTAINERS=$(podman ps -a --filter "network=my-ai-network" --format "{{.Names}}")

if [ -z "$CONTAINERS" ]; then
    output "No containers found in the 'my-ai-network' network."
else
    for container in $CONTAINERS; do
        output ""
        output "######################################################################"
        output "### LOGS FOR CONTAINER: $container"
        output "######################################################################"
        podman logs "$container" || output "--> Failed to retrieve logs for $container."
        output ""
    done
fi

output "Project Analysis Report - Generated: $(date)"
output "=================================================="

# Your existing command first
output "=================================================="
output "=============== PROJECT STRUCTURE =============="
output "=================================================="
# --- FIX #1: Removed '|volumes' to allow tree to scan the volumes directory ---
tree -I '.git|__pycache__|pip-cache'

output ""
output "=================================================="
output "================= FILE DETAILS ================="
output "=================================================="
# --- FIX #2: Removed the line that excluded the volumes directory from the find command ---
# --- FIX #3: Added an exclusion for dump.txt itself ---
find . \
  -path './.git' -prune -o \
  -path './dump.txt' -prune -o \
  -path '*/__pycache__' -prune -o \
  -path './volumes/pip-cache' -prune -o \
  -path './volumes/tts/xtts_v2_model/vocab.json' -prune -o \
  -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name 'Dockerfile' -o -name '*.yml' -o -name '*.md' -o -name '*.json' \) \
  -exec sh -c '
    echo "######################################################################"
    echo "### FILE: $1"
    echo "###-------------------------------------------------------------------"
    ls -lh "$1"
    echo "###-------------------------------------------------------------------"
    echo "### CONTENT:"
    cat "$1"
    echo
  ' sh {} \;

# ADDITIONAL CONTEXT GATHERING
output "=================================================="
output "============== DEPENDENCY ANALYSIS =============="
output "=================================================="

output "### Python Dependencies across all services:"
find . -name "*.py" -exec grep -l "^import\|^from" {} \; | while read file; do
    output "--- $file ---"
    grep -E "^(import|from)" "$file" | sort | uniq
    output ""
done

output "### Docker/Podman Network and Service Dependencies:"
output "Network references:"
grep -r "network\|--network" . --include="*.sh" --include="*.py" 2>/dev/null || output "None found"
output ""
output "Service URL references:"
grep -r "http://\|URL.*=" . --include="*.py" --include="*.sh" 2>/dev/null | grep -v ".git"
output ""

output "=================================================="
output "================ PORT MAPPING ==================="
output "=================================================="
output "### Exposed ports and service endpoints:"
grep -r "EXPOSE\|port.*=\|\.run.*port\|uvicorn.*port" . --include="Dockerfile" --include="*.py" --include="*.sh" 2>/dev/null
output ""

output "=================================================="
output "================ VOLUME MOUNTS =================="
output "=================================================="
output "### Volume and bind mount analysis:"
grep -r "\-v \|volume.*:" . --include="*.sh" 2>/dev/null || output "None found in scripts"
output ""

output "=================================================="
output "============= ENVIRONMENT VARIABLES ============="
output "=================================================="
output "### Environment variables used:"
grep -r "os\.getenv\|os\.environ\|\-e \|ENV " . --include="*.py" --include="Dockerfile" --include="*.sh" 2>/dev/null
output ""

output "=================================================="
output "=============== SERVICE FLOW ANALYSIS ==========="
output "=================================================="
output "### Service communication patterns:"
output "Services that make HTTP requests to other services:"
grep -r "requests\." . --include="*.py" -A 2 -B 1 2>/dev/null
output ""

output "=================================================="
output "================ GPU/HARDWARE USAGE ============="
output "=================================================="
output "### GPU and hardware requirements:"
grep -r "gpu\|cuda\|device.*=\|torch\." . --include="*.py" --include="Dockerfile" --include="*.sh" 2>/dev/null
output ""

output "=================================================="
output "=============== CONFIGURATION FILES ============="
output "=================================================="
output "### Looking for configuration files that might be missing from tree:"
find . -name "*.conf" -o -name "*.config" -o -name "*.ini" -o -name "*.env" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "requirements*.txt" -o -name "pyproject.toml" -o -name "setup.py" -o -name "poetry.lock" -o -name "Pipfile*" 2>/dev/null || output "None found"
output ""

output "=================================================="
output "================ ERROR HANDLING =================="
output "=================================================="
output "### Error handling patterns:"
grep -r "try:\|except\|raise\|HTTPException" . --include="*.py" 2>/dev/null | head -20
output "... (showing first 20 matches)"
output ""

output "=================================================="
output "================ MODEL PATHS & FILES ============"
output "=================================================="
output "### Model files and paths referenced:"
grep -r "model.*path\|MODEL_PATH\|\.gguf\|\.wav\|voices\|models" . --include="*.py" --include="*.sh" 2>/dev/null
output ""

output "=================================================="
output "================ CONTAINER ORCHESTRATION ========"
output "=================================================="
output "### Container management commands:"
grep -r "podman\|docker" . --include="*.sh" -A 1 -B 1 2>/dev/null
output ""

output "=================================================="
output "================ INCOMPLETE/TODO ITEMS =========="
output "=================================================="
output "### TODOs, FIXMEs, and placeholder content:"
grep -r -i "todo\|fixme\|placeholder\|# TODO\|# FIXME\|XXX\|HACK" . --include="*.py" --include="*.sh" --include="*.md" 2>/dev/null || output "None found"
output ""

output "=================================================="
output "================ PROJECT METADATA ==============="
output "=================================================="
output "### Git information (if available):"
if [ -d ".git" ]; then
    output "Git repository detected"
    git status --porcelain 2>/dev/null | head -10 || output "Git status unavailable"
    output "Recent commits:"
    git --no-pager log --oneline -5 2>/dev/null || output "Git log unavailable"
    output "Current branch:"
    git branch --show-current 2>/dev/null || output "Current branch unavailable"
else
    output "Not a git repository"
fi
output ""

output "### Directory sizes:"
du -sh */ 2>/dev/null | sort -hr
output ""

output "### File count by type:"
find . -type f | sed 's/.*\.//' | sort | uniq -c | sort -nr | head -10

output ""
output "=================================================="
output "Analysis complete!"
output "=================================================="
