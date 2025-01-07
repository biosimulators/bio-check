#!/usr/bin/env bash


# Run at root of repo!

argA="$1"

version=$(cat ./assets/docker/.BASE_VERSION)
img_name=ghcr.io/biosimulators/bio-compose-server-base:"$version"
latest_name=ghcr.io/biosimulators/bio-compose-server-base:latest

echo "Building base image..."
docker build --platform linux/amd64 -f ./Dockerfile -t "$img_name" .
echo "Built base image."

echo "Tagging new base image as latest..."
docker tag "$img_name" "$latest_name"

if [ "$argA" == "--push" ]; then
  # push version to GHCR
  docker push "$img_name"

  # push newest latest to GHCR
  docker push "$latest_name"
fi

