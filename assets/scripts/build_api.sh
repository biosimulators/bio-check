#!/usr/bin/env bash

set -e

version="$1"
gh_uname="$2"
implementation="$3"

lib=compose_api

if [ "$implementation" != "" ]; then
  export lib="$implementation"
fi

# build container
./assets/scripts/build_image.sh "$lib" "$version"

# push container
./assets/scripts/push_image.sh "$lib" "$version" "$gh_uname"
