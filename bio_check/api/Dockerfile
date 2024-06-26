 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

EXPOSE 3001

WORKDIR /app

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
    libhdf5-dev \
    pkg-config \
    wget \
    build-essential \
    cmake \
    swig \
    libbz2-dev

# Clean up the apt cache and remove build dependencies to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get remove -y build-essential && apt-get autoremove -y

ENV PYTHONUNBUFFERED 1

COPY ./requirements.api.txt ./assets/requirements.api.txt
COPY . /app/bio_check

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./assets/requirements.api.txt

CMD ["uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]
