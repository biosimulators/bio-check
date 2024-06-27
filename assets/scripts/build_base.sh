#!/usr/bin/env bash


# Run at root of repo!
set -e

prune="$1"  # --prune
version="$2"
push="$3"

if [ "$prune" == "--prune" ]; then
  docker system prune -a -f
fi

echo "Building base image..."
docker build -f ./Dockerfile-base -t ghcr.io/biosimulators/bio-check-base:"$version" .
echo "Built base image."

if [ "$push" == "--push" ]; then
  ./assets/scripts/push_base.sh "$version"
fi