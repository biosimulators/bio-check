#!/usr/bin/env bash

arg1="$1"
arg2="$2"

# optionally prune/clear system and cache prior to build
if [ "$arg1" == "-p" ]; then
  docker buildx prune -a -f
  docker system prune -a -f
fi

# remove old spec, create new
sudo rm compose_api/spec/openapi_3_1_0_generated.yaml
cd compose_api && python3 openapi_spec.py && cd ..

# remove pycache to clean images
sudo rm -r compose_api/__pycache__
sudo rm -r compose_worker/__pycache__

# build and push containers
docker compose build --no-cache

if [ "$arg2" == "-r" ]; then
  docker compose up
fi

if [ "$arg1" == "-r" ]; then
  docker compose up
fi

