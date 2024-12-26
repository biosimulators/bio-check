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

echo "Building base image..."
docker build --platform linux/amd64 -f ./Dockerfile-base -t compose-server-base:"$version" .
echo "Built base image."

echo "Tagging new base image as latest..."
docker tag compose-server-base:"$version" ghcr.io/biosimulators/compose-server-base:"$version"
docker tag ghcr.io/biosimulators/compose-server-base:"$version" ghcr.io/biosimulators/compose-server-base:latest
echo "New base image tagged:"
docker images | grep compose-server-base:latest

if [ "$argB" == "--push" ]; then
  # push version to GHCR
  docker push ghcr.io/biosimulators/compose-server-base:"$version"

  # push newest latest to GHCR
  docker push ghcr.io/biosimulators/compose-server-base:latest
else
  argB=""
fi

if [ "$argC" == "--run" ]; then
  ./assets/docker/scripts/run_container.sh base 1
fi
