#!/usr/bin/env bash

version="$1"

if [ "$version" == "" ]; then
  version="$(cat ./assets/BASE_VERSION.txt)"
fi

./assets/push_image.sh "base"
echo "$version" > ./assets/BASE_VERSION.txt