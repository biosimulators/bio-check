#!/usr/bin/env bash

set -e

lib="$1"
version="$2"
# gh_username="$3"
explicit_version=1

if [ "$version" == "" ]; then
  version=$(cat "$lib"/.VERSION)
  explicit_version=0
fi

# push version to GHCR
docker tag bio-compose-server-"$lib":"$version" ghcr.io/biosimulators/bio-compose-server-"$lib":"$version"
docker push ghcr.io/biosimulators/bio-compose-server-"$lib":"$version"

# push newest latest to GHCR
docker tag ghcr.io/biosimulators/bio-compose-server-"$lib":"$version" ghcr.io/biosimulators/bio-compose-server-"$lib":latest
docker push ghcr.io/biosimulators/bio-compose-server-"$lib":latest

# handle version
if [ "$lib" == "base" ]; then
  VERSION_FILE=./assets/.BASE_VERSION
else
  VERSION_FILE=./"$lib"/.VERSION
fi

if [ "$explicit_version" == 1 ]; then
  echo "Updating internal version of $lib with specified version."
  echo "$version" > "$VERSION_FILE"
fi

echo "Updated version: $version"
