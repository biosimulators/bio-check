FROM condaforge/miniforge3:latest
# FROM continuumio/miniconda3:main

LABEL org.opencontainers.image.title="bio-compose-server-base" \
    org.opencontainers.image.description="Base Docker image for BioCompose REST API management, job processing, and datastorage with MongoDB, ensuring scalable and robust performance." \
    org.opencontainers.image.url="https://compose.biosimulators.org/" \
    org.opencontainers.image.source="https://github.com/biosimulators/bio-compose-server" \
    org.opencontainers.image.authors="Alexander Patrie <apatrie@uchc.edu>, BioSimulators Team <info@biosimulators.org>" \
    org.opencontainers.image.vendor="BioSimulators Team"

ENV DEBIAN_FRONTEND=noninteractive

# copy assets
COPY assets/docker/config/.biosimulations.json /.google/.bio-compose.json
COPY assets/docker/config/.pys_usercfg.ini /Pysces/.pys_usercfg.ini
COPY assets/docker/config/.pys_usercfg.ini /root/Pysces/.pys_usercfg.ini
COPY tests/test_fixtures /test_fixtures

WORKDIR /bio-compose-server

# copy container libs
COPY ./gateway /bio-compose-server/gateway
COPY ./shared /bio-compose-server/shared
COPY ./worker /bio-compose-server/worker

# copy env configs
COPY ./environment.yml /bio-compose-server/environment.yml
COPY ./pyproject.toml /bio-compose-server/pyproject.toml
RUN echo "Server" > /bio-compose-server/README.md

RUN mkdir config \
    && apt-get update \
    && apt-get install -y \
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
        libstdc++6

RUN conda update -n base -c conda-forge conda \
    && conda env create -f /bio-compose-server/environment.yml -y \
    && conda run -n server pip install --upgrade pip \
    && echo "conda activate server" >> /.bashrc

RUN conda run -n server poetry config virtualenvs.create false \
    && conda run -n server poetry lock \
    && conda run -n server poetry install

# && conda config --env --add channels conda-forge \
# && conda config --set channel_priority strict \
# && conda install readdy \
# && conda env create -f environment.yml -y
# && conda install -c conda-forge pymem3dg -y \
# && echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH' >> ~/.bashrc

# expose for gateway
EXPOSE 3001
