#!/usr/bin/env bash

set -e

version="$1"
gh_username="$2"

if [ "$version" == "" ]; then
  version="$(cat ./assets/BASE_VERSION.txt)"
fi

./assets/scripts/push_image.sh "base" "$version" "$gh_username"
echo "$version" > ./assets/BASE_VERSION.txt