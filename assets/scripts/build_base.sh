#!/usr/bin/env bash


# Run at root of repo!
set -e

version="$1"
argA="$2"
argB="$3"  # -p
argC="$4"

if [ "$argA" == "-p" ] || [ "$argB" == "-p" ] || [ "$argC" == "-p" ]; then
  docker system prune -f -a
fi

echo "Building base image..."
docker build --platform linux/amd64 -f ./Dockerfile-base -t ghcr.io/biosimulators/bio-check-base:"$version" .
echo "Built base image."

echo "Tagging new base image as latest..."
docker tag ghcr.io/biosimulators/bio-check-base:"$version" ghcr.io/biosimulators/bio-check-base:latest
echo "New base image tagged:"
docker images

if [ "$argA" == "--push" ] || [ "$argB" == "--push" ] || [ "$argC" == "--push" ]; then
  ./assets/scripts/push_base.sh "$version"
fi

if [ "$argA" == "--run" ] || [ "$argB" == "--run" ] || [ "$argC" == "--run" ]; then
  ./assets/scripts/run_container.sh base latest
fi