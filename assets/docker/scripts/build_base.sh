#!/usr/bin/env bash


# Run at root of repo!
set -e

argA="$1"
argB="$2"
argC="$3"

version=$(cat ./assets/docker/.BASE_VERSION)

if [ "$argA" == "--prune" ] || [ "$argB" == "--prune" ] || [ "$argC" == "--prune" ]; then
  docker system prune -f -a
fi

echo "Building base image..."
docker build --platform linux/amd64 -f ./Dockerfile-base -t bio-check-base:"$version" .
echo "Built base image."

echo "Tagging new base image as latest..."
docker tag bio-check-base:"$version" bio-check-base:latest
echo "New base image tagged:"
docker images

if [ "$argB" == "--push" ] || [ "$argA" == "--push" ] || [ "$argC" == "--push" ]; then
  # push version to GHCR
  docker tag bio-check-base:"$version" ghcr.io/biosimulators/bio-check-base:"$version"
  docker push ghcr.io/biosimulators/bio-check-base:"$version"

  # push newest latest to GHCR
  docker tag ghcr.io/biosimulators/bio-check-base:"$version" ghcr.io/biosimulators/bio-check-base:latest
  docker push ghcr.io/biosimulators/bio-check-base:latest
fi

if [ "$argB" == "--run" ] || [ "$argA" == "--run" ] || [ "$argC" == "--run" ]; then
  ./assets/docker/scripts/run_container.sh
