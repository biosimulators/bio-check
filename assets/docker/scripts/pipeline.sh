#!/usr/bin/env bash


function run_biocom_pipeline {
  deploy="@1"
  no_build_all="@2"

  # build and deploy base
  echo "Building base image..."
  if /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/build_base.sh; then
    echo "Successfully built new base image. Currently installed docker images:"
    docker images
  else
    echo "Exiting..."
    exit 1
  fi
  echo "Deploying base..."
  /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/push_base.sh
  echo "Successfully deployed base image."

  if [ "$no_build_all" != "--no-build-all" ]; then
    # build and deploy microservices
    /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/build_microservices.sh
    set -e
    if [ "$deploy" == "-d" ]; then
      echo "Deploying API microservice..."
      /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/push_image.sh gateway

      echo "Deploying Worker microservice..."
      /Users/alexanderpatrie/desktop/repos/bio-check/assets/docker/scripts/push_image.sh worker

      echo "Images successfully deployed."
    fi
  fi
}

function run_cd {
  deploy="@1"
  build_all="@2"

  if [ "$deploy" == "-d" ]; then
    echo "Deployment mode on!"
  else
    echo "Deployment mode off!"
  fi

  docker system prune -a -f
  run_biocom_pipeline "$deploy" "$build_all"
}


run_cd





