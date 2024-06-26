FROM python:3.11-slim-bullseye
LABEL authors="alexanderpatrie"

RUN mkdir /app
WORKDIR /app

COPY docker-assets ./assets

ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

# hdf5 deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    linux-headers-arm64 \
    libjpeg-dev \
    zlib1g-dev \
    libmariadb-dev \
    libffi-dev \
    libopenblas-dev \
    libgfortran5 \
    liblapack-dev \
    build-essential \
    libssl-dev \
    libhdf5-dev

# Clean up the apt cache and remove build dependencies to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get remove -y build-essential && apt-get autoremove -y

ENV PYTHONUNBUFFERED 1

COPY . /app/

CMD ["pip", "install", "-r", "--no-cache-dir", "./assets/requirements.base.txt"]
