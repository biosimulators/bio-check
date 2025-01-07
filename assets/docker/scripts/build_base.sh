#!/usr/bin/env bash


# Run at root of repo!

argA="$1"
argB="$2"
argC="$3"

version=$(cat ./assets/docker/.BASE_VERSION)

if [ "$argA" == "--prune" ]; then
  docker system prune -f -a
else
  argA=""
fi

img_name=ghcr.io/biosimulators/bio-compose-server-base:"$version"
latest_name=ghcr.io/biosimulators/bio-compose-server-base:latest

echo "Building base image..."
docker build --platform linux/amd64 -f ./Dockerfile -t "$img_name" .
echo "Built base image."

echo "Tagging new base image as latest..."
docker tag "$img_name" "$latest_name"

if [ "$argB" == "--push" ]; then
  # push version to GHCR
  docker push "$img_name"

  # push newest latest to GHCR
  docker push "$latest_name"
else
  argB=""
fi

if [ "$argC" == "--run" ]; then
  ./assets/docker/scripts/run_container.sh base 1
fi
