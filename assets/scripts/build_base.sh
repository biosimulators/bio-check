#!/usr/bin/env bash


# Run at root of repo!
set -e

version="$1"
push="$2"
prune="$3"  # --prune

if [ "$prune" == "--prune" ] || [ "$push" == "--prune" ]; then
  docker system prune -f -a
fi

echo "Building base image..."
docker build --platform linux/amd64 -f ./Dockerfile-base -t ghcr.io/biosimulators/bio-check-base:"$version" .
echo "Built base image."

echo "Tagging new base image as latest..."
docker tag ghcr.io/biosimulators/bio-check-base:"$version" ghcr.io/biosimulators/bio-check-base:latest
echo "New base image tagged:"
docker images

if [ "$push" == "--push" ]; then
  ./assets/scripts/push_base.sh "$version"
fi