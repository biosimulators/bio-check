FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.title="bio-compose-server-base" \
    org.opencontainers.image.description="Base Docker image for BioCompose REST API management, job processing, and datastorage with MongoDB, ensuring scalable and robust performance." \
    org.opencontainers.image.url="https://compose.biosimulators.org/" \
    org.opencontainers.image.source="https://github.com/biosimulators/bio-compose-server" \
    org.opencontainers.image.authors="Alexander Patrie <apatrie@uchc.edu>, BioSimulators Team <info@biosimulators.org>" \
    org.opencontainers.image.vendor="BioSimulators Team"

ENV DEBIAN_FRONTEND=noninteractive

# copy assets
COPY assets/docker/config/.pys_usercfg.ini /Pysces/.pys_usercfg.ini
COPY assets/docker/config/.pys_usercfg.ini /root/Pysces/.pys_usercfg.ini
COPY tests/test_fixtures /test_fixtures

WORKDIR /app

# copy container libs
COPY . .

# copy env configs
COPY ./environment.yml /app/environment.yml
COPY ./pyproject.toml /app/pyproject.toml


# install deps
RUN mkdir config \
    && apt-get update  \
    && apt install -y \
    meson \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libgfortran5 \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    cmake \
    make \
    git \
    build-essential \
    python3-dev \
    swig \
    libc6-dev \
    libx11-dev \
    libc6 \
    libgl1-mesa-dev \
    pkg-config \
    curl \
    tar \
    libgl1-mesa-glx \
    libice6 \
    libsm6 \
    gnupg \
    nano \
    libstdc++6 \
    && conda update -n base -c conda-forge conda \
    && conda env create -f environment.yml -y

# && echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc

# expose for gateway
EXPOSE 3001
