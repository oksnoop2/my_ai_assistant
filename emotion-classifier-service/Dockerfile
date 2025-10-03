# in emotion-classifier-service/Dockerfile

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install pip and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install the required python libraries.
# We install the CPU version of PyTorch as this model is small and fast on CPU.
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    "transformers[torch]" \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
