#!/usr/bin/env bash

set -e

lib="$1"
version="$2"
gh_username="$3"

if [ "$version" == "" ]; then
  echo "You must pass a version as second arg."
  exit 1
fi

# if [ "$version" == "$current" ]; then
#   echo "This version already exists on GHCR. Exiting."
#   exit 1
# fi

# login to github
./assets/scripts/gh_login.sh "$gh_username"

# yes | docker system prune

# push version
docker push ghcr.io/biosimulators/bio-check-"$lib":"$version"

# tag version as latest
docker tag ghcr.io/biosimulators/bio-check-"$lib":"$version" ghcr.io/biosimulators/bio-check-"$lib":latest

# push newest latest
docker push ghcr.io/biosimulators/bio-check-"$lib":latest

echo "$version" > ./"$lib"/CONTAINER_VERSION.txt