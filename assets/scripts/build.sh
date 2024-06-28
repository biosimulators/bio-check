#!/usr/bin/env bash

export PKG_ROOT="ghcr.io/biosimulators/bio-check"
version="$1"

# clear system
yes | docker system prune -a
yes | docker buildx prune


# build base
docker build -f ./Dockerfile-base -it "$PKG_ROOT"-base:"$version" .

# build api
docker build -f bio_check/api/Dockerfile-api -t "$PKG_ROOT"-api:"$version" ./bio_check/api

# build worker
docker build -f bio_check/worker/Dockerfile-worker -t "$PKG_ROOT"-worker:"$version" ./bio_check/worker
