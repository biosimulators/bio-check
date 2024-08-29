#!/usr/bin/env bash

function build_container {
  lib=$1
  img=$2
  # Build the Docker image
  docker build --no-cache --platform linux/amd64 -f ./"$lib"/Dockerfile-"$lib" -t "$img" ./"$lib"
}

function push_container {
  lib=$1
  version=$2

  echo "GitHub UserName: "
  read -r gh_username

  # login to github
  ./assets/scripts/gh_login.sh "$gh_username"

  # push version
  docker push ghcr.io/biosimulators/bio-check-"$lib":"$version"

  # tag version as latest
  docker tag ghcr.io/biosimulators/bio-check-"$lib":"$version" ghcr.io/biosimulators/bio-check-"$lib":latest

  # push newest latest
  docker push ghcr.io/biosimulators/bio-check-"$lib":latest

  # handle version
  working_dir="$(pwd)"
  if [ "$lib" == "base" ]; then
    VERSION_FILE="$working_dir/assets/.BASE_VERSION"
  else
    VERSION_FILE="$working_dir/$lib/.CONTAINER_VERSION"
  fi
  echo "$version" > "$VERSION_FILE"
  echo "Updated version: $version"
}

function deploy_container {
  echo "Library to build: "
  read -r lib

  echo "Organization of container: "
  read -r org

  echo "Container name: "
  read -r container_name

  if [[ "$container_name" != *:* ]]; then
    echo "Container version not specified. Please enter the version: "
    read -r version
    container_name="${container_name}:${version}"
  else
    version="${container_name##*:}"
  fi

  img="ghcr.io/$org/$container_name"

  # build img
  build_container "$lib" "$img"

  # deploy
  push_container "$lib" "$version"
}




