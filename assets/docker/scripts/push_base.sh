#!/usr/bin/env bash

set -e

version="$1"
gh_username="$2"
explicit_version=1

if [ "$version" == "" ]; then
  version="$(cat ./assets/docker/.BASE_VERSION)"
  explicit_version=0
fi

./assets/docker/scripts/push_image.sh "base" "$version" "$gh_username"

# update base if version is specified
if [ "$explicit_version" == 1 ]; then
  echo "$version" > ./assets/docker/.BASE_VERSION
fi