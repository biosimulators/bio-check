#!/usr/bin/env bash


# Run at root of repo!
set -e

prune="$1"  # --prune
push="$2"
version="$3"

if [ "$prune" == "--prune" ]; then
  docker system prune -a -f
fi

echo "Building base image..."
docker build -f ./Dockerfile-base -t ghcr.io/biosimulators/bio-check-base .
echo "Built base image."

if [ "$push" == "--push" ]; then
  ./assets/scripts/push_base.sh "$version"
fi