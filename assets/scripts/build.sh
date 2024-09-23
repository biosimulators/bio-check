#!/usr/bin/env bash

# TO BE RUN AT REPO ROOT
# args = -a (build/push all) OR compose_api-OR-compose_worker
# kwargs = -d (deploy to k8 after push)

arg="$1"
deploy="$2"

set -e

# clear system
yes | docker system prune -a && yes | docker buildx prune -a

# clear pycaches
python3 ./assets/scripts/rm_pycache.py

# build/push images
if [ "$arg" != "-a" ]; then
  # build/push single image
  lib=arg
  docker compose build "$lib" --no-cache
  version=$(python3 ./.github/parse_container_version.py "$lib")
  ./assets/scripts/push_image.sh "$lib" "$version" AlexPatrie
else
  # build/push all
  docker compose build --no-cache
  echo "Docker images built!"

  api_version=$(python3 ./.github/parse_container_version.py compose_api)
  echo "Using API version: $api_version"

  worker_version=$(python3 ./.github/parse_container_version.py compose_worker)
  echo "Using worker version: $worker_version"

  ./assets/scripts/push_image.sh compose_api "$api_version" AlexPatrie
  ./assets/scripts/push_image.sh compose_worker "$worker_version" AlexPatrie
fi

# optionally deploy to k8
if [ "$deploy" == "-d" ]; then
  cd kustomize \
  && kubectl kustomize overlays/biochecknet | kubectl apply -f - \
  && cd ..
  echo "Overlays applied!"
fi
