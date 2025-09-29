# my_ai_assistant/llm/Dockerfile (Corrected and Final Version)

FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime AND build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgomp1 \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Python libraries for the service, specifically targeting CUDA 12.4
RUN pip3 install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    requests \
    llama-cpp-python \
    torch --extra-index-url https://download.pytorch.org/whl/cu124

EXPOSE 8080
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
