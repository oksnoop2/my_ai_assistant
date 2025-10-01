# commit-helper-service/Dockerfile
#boop
FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Add this to ensure Python output is not buffered, which is better for container logging.
ENV PYTHONUNBUFFERED=1

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

# FIX: Use python3.11's pip explicitly to install packages.
RUN python3.11 -m pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    requests \
    llama-cpp-python \
    torch --extra-index-url https://download.pytorch.org/whl/cu124

EXPOSE 8080

# FIX: Run uvicorn using the python3.11 module syntax.
CMD ["python3.11", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
