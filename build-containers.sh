#!/usr/bin/env bash 

set -e 

PKG_ROOT=ghcr.io/biosimulators/bio-check
base_version="$1"
pkg_version="$2"

                           
docker build -f ./Dockerfile-base -t "$PKG_ROOT"-base:"$base_version" .
docker tag "$PKG_ROOT"-base:"$base_version" "$PKG_ROOT"-base:latest
docker build -f ./api/Dockerfile-api -t ghcr.io/biosimulators/bio-check-api:0.0.0 ./api
docker build -f ./worker/Dockerfile-worker -t ghcr.io/biosimulators/bio-check-worker:0.0.0 ./worker