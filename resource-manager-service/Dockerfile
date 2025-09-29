# resource-manager/Dockerfile
FROM python:3.11-slim

RUN pip3 install --cache-dir /cache \
    fastapi \
    uvicorn \ 
    requests

WORKDIR /app
COPY . .

EXPOSE 8080
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
