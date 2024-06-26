#!/usr/bin/env bash

lib="$1"
version="$2"

# PLEASE UPDATE THE LATEST VERSION HERE BEFORE RUNNING. CURRENT: 0.0.4
current="0.0.0"

if [ "$version" == "" ]; then
  echo "You must pass the container version you wish to release as an argument to this script. Exiting."
  exit 1
fi

if [ "$version" == "$current" ]; then
  echo "This version already exists on GHCR. Exiting."
  exit 1
fi

echo "Enter your Github User-Name: "
read -r usr_name

if docker login ghcr.io -u "$usr_name"; then
  echo "Successfully logged in to GHCR!"
else
  echo "Could not validate credentials."
  exit 1
fi

# yes | docker system prune

# push version
docker push ghcr.io/biosimulators/bio-check-"$lib":"$version"

# tag version as latest
docker tag ghcr.io/biosimulators/bio-check-"$lib":"$version" ghcr.io/biosimulators/bio-check-"$lib":latest

# push newest latest
docker push ghcr.io/biosimulators/bio-check-"$lib":latest