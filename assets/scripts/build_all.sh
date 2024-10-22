#!/usr/bin/env bash

set -e

arg1="$1"

# optionally prune/clear system and cache prior to build
if [ "$arg1" == "-p" ]; then
  yes | docker system prune -a
  yes | docker buildx prune -a
fi

# remove pycache to clean images
sudo rm -r compose_api/__pycache__
sudo rm -r compose_worker/__pycache__

# build and push containers
docker compose build --no-cache

