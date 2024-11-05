#!/usr/bin/env bash

set -e

lib="$1"
version="$2"
# gh_username="$3"

if [ "$version" == "" ]; then
  version=$(cat "$lib"/.VERSION)
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

echo "$version" > "$VERSION_FILE"

# Optional: Output the new version
echo "Updated version: $version"
