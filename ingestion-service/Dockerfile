# in ./ingestion-service/Dockerfile

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies. python-magic-bin requires libmagic1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Install all necessary Python libraries for parsing and communication
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    ollama \
    trafilatura \
    beautifulsoup4 \
    lxml \
    tqdm \
    PyPDF2 \
    python-docx \
    llama-index-core \
    python-magic

EXPOSE 8007

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8007"]
