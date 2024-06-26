FROM python:3.11-alpine

RUN mkdir /app
WORKDIR /app

COPY ./dockerfile-assets ./assets

ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

# hdf5 deps
RUN apk add --update --no-cache --virtual .tmp gcc libc-dev linux-headers
RUN apk add --no-cache jpeg-dev zlib-dev mariadb-dev libffi-dev openblas-dev libgfortran lapack-dev build-base openssl-dev
RUN apk add --no-cache hdf5-dev
RUN pip install -r /requirements.txt
RUN apk --no-cache del build-base

ENV PYTHONUNBUFFERED 1

COPY . /app/

CMD ["pip", "install", "-r", "--no-cache-dir", "./assets/requirements.base.txt"]

# FROM continuumio/miniconda3
# ADD dockerfile-assets/environment.yml /tmp/environment.yml
# RUN conda env create -f /tmp/environment.yml
# # Pull the environment name out of the environment.yml
# RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
# CMD ["bash"]

# SHELL ["conda", "run", "-n", "bio-check", "/bin/bash", "-c"]

# CMD ["conda", "run", "-n", "bio-check", "python", "your_script.py"]

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.10  \
#     ca-certificates \
#     python3-pip  \
#     python3-dev \
#     build-essential  \
#     libncurses5  \
#     cmake  \
#     make  \
#     libx11-dev  \
#     libc6-dev  \
#     libx11-6  \
#     libc6  \
#     gcc  \
#     swig \
#     pkg-config  \
#     curl  \
#     tar  \
#     libgl1-mesa-glx  \
#     libice6  \
#     libpython3.10  \
#     libsm6 \
#     wget  \
#     && rm -rf /var/lib/apt/lists/* \
#     && pip install --upgrade pip \
#     && apt-get clean \
#     && apt-get autoclean \
#     && pip install --no-cache-dir -r ./dockerfile-assets/requirements.base.txt

# 1.
# docker system prune -a -f && \

# 2.
# sudo docker build -t spacebearamadeus/bio-check-base . && \
# sudo docker build -t spacebearamadeus/bio-check-api ./bio_check/api && \

# 3.
# docker run -d -p 8000:3001 spacebearamadeus/bio-check-api
            # OR
# docker run -it -p 8000:3001 spacebearamadeus/bio-check-api

# docker system prune -a -f && \
# sudo docker build -t spacebearamadeus/bio-check-base . && \
# sudo docker build -t spacebearamadeus/bio-check-api ./bio_check/api && \
# docker run --platform linux/amd64 -it -p 8000:3001 spacebearamadeus/bio-check-api