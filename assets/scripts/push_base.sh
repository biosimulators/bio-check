#!/usr/bin/env bash

set -e

version="$1"

if [ "$version" == "" ]; then
  version="$(cat ./assets/BASE_VERSION.txt)"
fi

./assets/scripts/push_image.sh "base"
echo "$version" > ./assets/BASE_VERSION.txt