#!/usr/bin/env bash

# Run at root of repo!

lib="$1"
build_base="$2"

docker system prune -a -f

if [ -n "$build_base" ]; then
  echo "Building base image..."
  sudo docker build -t ghcr.io/biosimulators/bio-check-base .
  echo "Built base image."
fi

cd bio_check/"$lib" || exit
sudo docker build -t ghcr.io/biosimulators/bio-check-"$lib" ./bio_check/"$lib"
docker run --platform linux/amd64 -it -p 8000:3001 ghcr.io/biosimulators/bio-check-"$lib"