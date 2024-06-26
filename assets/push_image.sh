#!/usr/bin/env bash

lib="$1"
version="$2"

if [ "$version" == "" ]; then
  version="latest"
fi

# if [ "$version" == "$current" ]; then
#   echo "This version already exists on GHCR. Exiting."
#   exit 1
# fi

# login to github
./gh_login.sh

# yes | docker system prune

# push version
docker push ghcr.io/biosimulators/bio-check-"$lib":"$version"

# tag version as latest
docker tag ghcr.io/biosimulators/bio-check-"$lib":"$version" ghcr.io/biosimulators/bio-check-"$lib":latest

# push newest latest
docker push ghcr.io/biosimulators/bio-check-"$lib":latest