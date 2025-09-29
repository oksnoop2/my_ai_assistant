# ./tts-service/Dockerfile
# tells Podman to build your image on top of a foundation provided by NVIDIA. This base image already contains the Ubuntu operating system, the necessary CUDA drivers, and libraries (cudnn) that PyTorch will need to communicate with the GPU.
FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04


# This RUN instruction executes a series of commands inside the container to install software at the operating system level. After the installation is done, it deletes the package lists that were downloaded by apt-get update. Because this is done in the same RUN command, it reduces the size of this specific layer in the final image.

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# This command uses the version of pip that came with the base image to upgrade itself (pip), along with its core dependencies (setuptools, wheel), to their latest versions.
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
#creates a directory named /app inside the container if it doesn't already exist and sets this directory as the current location for all subsequent commands (RUN, COPY, CMD). like running cd /app in a terminal.
WORKDIR /app
# The first . means "the current directory on the host machine" (i.e., ./tts-service/). The second . means "the current working directory inside the container" (which we just set to /app). line copies service.py and your Dockerfile into the /app/ directory inside the container.
COPY . .
#     pip3 install ...: Tells pip to install all the listed Python libraries. --cache-dir /cache: tells pip to use the /cache directory for its package cache. Because you started your build with podman build --volume $PWD/volumes/pip-cache:/cache:z, pip is actually using your persistent host directory. It will pull any existing .whl files from there and save any new downloads there for future builds.
RUN pip3 install --cache-dir /cache \
    fastapi \
    uvicorn \
    TTS \
    numpy \
    torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 \
    transformers==4.35.0
#exposes the container port. This command is actually just documentation. It doesn't publish the port itself. It's a piece of metadata that signals to the user (and other tools) that the service running inside this container is intended to listen on port 8000. The actual mapping of the host port to the container port happens in your podman run command with the -p flag (e.g., -p 8002:8000).
EXPOSE 8000
#    "uvicorn": This is the program to execute. uvicorn is a web server that runs your Python web application. "service:app": This tells uvicorn what code to run. It means: "Look for a file named service.py, and inside that file, find the FastAPI object named app." "--host", "0.0.0.0": This is critical for containers. It tells the server to listen for traffic on all network interfaces inside the container. If you used the default (127.0.0.1 or localhost), it would only accept connections from within the container itself, and you wouldn't be able to connect from your host machine. "--port", "8000": This tells uvicorn to listen on port 8000 inside the container. This must match the port you used in the EXPOSE instruction.
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
