#!/usr/bin/env bash

set -e

lib="$1"
version="$2"
# gh_username="$3"

if [ "$version" == "" ]; then
  echo "You must pass a version as second arg."
  exit 1
fi

# login to github
# ./assets/scripts/gh_login.sh "$gh_username"

# yes | docker system prune

# push version
docker push ghcr.io/biosimulators/bio-check-"$lib":"$version"

# tag version as latest
docker tag ghcr.io/biosimulators/bio-check-"$lib":"$version" ghcr.io/biosimulators/bio-check-"$lib":latest

# push newest latest
docker push ghcr.io/biosimulators/bio-check-"$lib":latest

# handle version
if [ "$lib" == "base" ]; then
  VERSION_FILE=./assets/.BASE_VERSION
else
  VERSION_FILE=./"$lib"/.CONTAINER_VERSION
fi

echo "$version" > "$VERSION_FILE"

# Optional: Output the new version
echo "Updated version: $version"
