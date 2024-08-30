#!/usr/bin/env bash

set -e

version="$1"
arg1="$2"
un=AlexPatrie

# optionally prune/clear system and cache prior to build
if [ "$arg1" == "-p" ]; then
  yes | docker system prune -a
fi

# build and push containers
./assets/scripts/build_api.sh "$version" "$un" && ./assets/scripts/build_worker.sh "$version" "$un"

# optionally apply changes to container versions in Kustomization
if [ "$arg1" == "-k" ]; then
  cd kustomize \
    && kubectl kustomize overlays/biochecknet | kubectl apply -f - \
    && cd ..
fi