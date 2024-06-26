#!/usr/bin/env bash

# Args:
# $1: the library for which you would like to build a container
# -p ($2): whether to prune docker system before building anything.
# -b ($3): whether to build the base image before building library
# Run at root of repo!

lib="$1"
prune="$2"  # -p
build_="$3"  # -b

if [ "$prune" ]; then
  docker system prune -a -f
fi

if [ "$build_" ]; then
  echo "Building base image..."
  sudo docker build -t ghcr.io/biosimulators/bio-check-base .
  echo "Built base image."
fi

cd bio_check/"$lib" || exit
sudo docker build -t ghcr.io/biosimulators/bio-check-"$lib" ./bio_check/"$lib"
docker run --platform linux/amd64 -it -p 8000:3001 ghcr.io/biosimulators/bio-check-"$lib"
