#!/usr/bin/env bash

export PKG_ROOT=ghcr.io/biosimulators/bio-check
version="$1"
base_version="$2"

# clear system
yes | docker system prune -a

# build base
docker build -f ./Dockerfile-base -t "$PKG_ROOT"-base:"$base_version" .

# build api
docker build -f bio_check/api/Dockerfile-api -t "$PKG_ROOT"-api:"$version" ./bio_check/api

# build worker
docker build -f bio_check/worker/Dockerfile-worker -t "$PKG_ROOT"-worker:"$version" ./bio_check/worker
