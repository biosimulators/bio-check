FROM python:3.10-slim-buster

LABEL authors="alexanderpatrie"

WORKDIR /src

COPY verification_service ./verification_service
COPY dockerfile-assets ./dockerfile-assets

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential  \
    libncurses5  \
    bash \
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
    libsm6 \
    wget  \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && apt-get clean \
    && apt-get autoclean \
    && pip install --no-cache-dir -r ./dockerfile-assets/requirements-base.txt

CMD ["bash"]

# docker system prune -a -f && \
# sudo docker build -t spacebearamadeus/verification-service-base . && \
# sudo docker build -t spacebearamadeus/verification-service-api ./verification_service/api && \
# docker run -d -p 8000:3001 spacebearamadeus/verification-service-api