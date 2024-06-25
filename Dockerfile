FROM python:3.10-slim-buster
# FROM ghcr.io/biosimulators/biosimulators:latest
LABEL authors="alexanderpatrie"

WORKDIR /src

RUN apt-get update -y \
    && apt-get install --no-install-recommends -y \
        wget \
        build-essential \
        cmake \
        swig \
        libbz2-dev

COPY verification_service ./verification_service

COPY dockerfile-assets ./dockerfile-assets

RUN pip install --no-cache-dir -r ./dockerfile-assets/requirements-base.txt

# RUN pip install --upgrade pip

# docker system prune -a -f && \
# sudo docker build -t spacebearamadeus/verification-service-base . && \
# sudo docker build -t spacebearamadeus/verification-service-api ./verification_service/api && \
# docker run -d -p 8000:3001 spacebearamadeus/verification-service-api