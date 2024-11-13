FROM tensorflow/tensorflow:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
