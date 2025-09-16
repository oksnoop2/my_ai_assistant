# my_ai_assistant/llm/Dockerfile (Correct and Final Version)

# Start from the official NVIDIA CUDA runtime image, which provides CUDA 12.1
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies: Python, pip, AND the missing OpenMP library.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install Python libraries for the service
RUN pip3 install --no-cache-dir fastapi uvicorn "uvicorn[standard]"

# Install the pre-built llama-cpp-python wheel with CUDA 12.1 support.
RUN pip3 install --no-cache-dir llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

EXPOSE 8080
CMD ["uvicorn", "llm_service:app", "--host", "0.0.0.0", "--port", "8080"]
