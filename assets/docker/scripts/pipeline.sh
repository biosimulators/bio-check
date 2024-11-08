#!/usr/bin/env bash


function run_biocom_pipeline {
  deploy="@1"

  echo "Building base image..."
  /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/build_base.sh

  echo "Successfully built new base image. Currently installed docker images:"
  docker images

  echo "Deploying base..."
  /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/push_base.sh
  echo "Successfully deployed base image."

  /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/build_microservices.sh

  set -e
  if [ "$deploy" == "-d" ]; then
    echo "Deploying API microservice..."
    /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/push_image.sh api

    echo "Deploying Worker microservice..."
    /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/push_image.sh worker

    echo "Images successfully deployed."
  fi
}

function biocom_cd {
  deploy="@1"

  docker system prune -a -f
  run_biocom_pipeline "$deploy"
}







