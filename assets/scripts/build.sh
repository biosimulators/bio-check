#!/usr/bin/env bash

export PKG_ROOT=ghcr.io/biosimulators/bio-check
lib_version="$1"
base_version="$2"

# clear system
yes | docker system prune -a

# build base
docker build -f ./Dockerfile-base -t "$PKG_ROOT"-base:"$base_version" .

# tag latest for base
docker tag "$PKG_ROOT"-base:"$base_version" "$PKG_ROOT"-base:latest

# build api
docker build -f bio_check/api/Dockerfile-api -t "$PKG_ROOT"-api:"$lib_version" ./bio_check/api

# build worker
docker build -f bio_check/worker/Dockerfile-worker -t "$PKG_ROOT"-worker:"$lib_version" ./bio_check/worker
