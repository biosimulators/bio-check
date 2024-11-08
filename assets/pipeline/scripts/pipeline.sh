#!/usr/bin/env bash

arg1="$1"

docker system prune -a -f

echo "Building base image..."
./assets/docker/scripts/build_base.sh

echo "Successfully built new base image. Currently installed docker images:"
docker images

if [ "$arg1" == "-d" ]; then
  echo "Deploying base..."
  ./assets/docker/scripts/push_base.sh
  echo "Successfully deployed base image."
fi

./assets/pipeline/scripts/build_microservices.sh

set -e
if [ "$arg1" == "-d" ]; then
  echo "Deploying API microservice..."
  ./assets/docker/scripts/push_image.sh api

  echo "Deploying Worker microservice..."
  ./assets/docker/scripts/push_image.sh worker

  echo "Images successfully deployed."
fi

