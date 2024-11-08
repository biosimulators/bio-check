#!/usr/bin/env bash

prune="$1"
lib="$2"

if [ "$prune" == "-p" ]; then
  docker system prune -a -f
fi

# remove old spec and create new spec
rm -f api/spec/openapi_3_1_0_generated.yaml
conda run -n server python api/openapi_spec.py

# remove pycaches if exists
sudo rm -r api/__pycache__
sudo rm -r worker/__pycache__

if [ "$lib" != "" ]; then
  echo "Building specified library: $lib"
  docker compose build "$lib" --no-cache
  echo "$lib built!"
else
  echo "No specific library specified."
  echo "Building API microservice..."
  if docker compose build api --no-cache; then
    echo "API built!"
  else
    echo "API build failed. Now attempting to build Worker..."
  fi

  echo "Building Worker microservice..."
  if docker compose build worker --no-cache; then
    echo "Worker built!"
  else
    echo "Worker build failed."
    exit 1
  fi
fi