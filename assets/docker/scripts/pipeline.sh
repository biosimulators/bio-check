#!/usr/bin/env bash

arg1="$1"
arg2="$2"

if [ "$arg1" == "-p" ] || [ "$arg2" == "-p" ]; then
  docker system prune -a -f
fi

echo "Building base image..."
./assets/docker/scripts/build_base.sh
echo "Successfully built new base image. Currently installed docker images:"
docker images

echo "Building microservices..."
docker compose build --no-cache
echo "Successfully built new microservice images. Currently installed docker images:"
docker images

if [ "$arg1" == "-d" ] || [ "$arg2" == "-d" ]; then
  set -e

  echo "Deploying base..."
  ./assets/docker/scripts/push_base.sh
  echo "Successfully deployed base image."

  echo "Deploying API microservice..."
  ./assets/docker/scripts/push_image.sh api

  echo "Deploying Worker microservice..."
  ./assets/docker/scripts/push_image.sh worker

  echo "Images successfully deployed."
fi


