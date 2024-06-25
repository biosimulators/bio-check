FROM ubuntu:22.04

LABEL authors="alexanderpatrie"

WORKDIR /src

COPY verification_service ./verification_service
COPY dockerfile-assets ./dockerfile-assets

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10  \
    ca-certificates \
    python3-pip  \
    python3-dev \
    build-essential  \
    libncurses5  \
    cmake  \
    make  \
    libx11-dev  \
    libc6-dev  \
    libx11-6  \
    libc6  \
    gcc  \
    swig \
    pkg-config  \
    curl  \
    tar  \
    libgl1-mesa-glx  \
    libice6  \
    libpython3.10  \
    libsm6 \
    wget  \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && apt-get clean \
    && apt-get autoclean \
    && pip install --no-cache-dir -r ./dockerfile-assets/requirements-base.txt

# 1.
# docker system prune -a -f && \

# 2.
# sudo docker build -t spacebearamadeus/verification-service-base . && \
# sudo docker build -t spacebearamadeus/verification-service-api ./verification_service/api && \

# 3.
# docker run -d -p 8000:3001 spacebearamadeus/verification-service-api
            # OR
# docker run -it -p 8000:3001 spacebearamadeus/verification-service-api

# docker system prune -a -f && \
# sudo docker build -t spacebearamadeus/verification-service-base . && \
# sudo docker build -t spacebearamadeus/verification-service-api ./verification_service/api