#!/usr/bin/env bash

# Args:
# $1: the library for which you would like to build a container
# --run ($2): whether to run the library container after build
# --prune ($3): whether to prune docker system before building anything.
# Run at root of repo!

lib="$1"
run_="$2" # --run
prune="$3"  # --prune

set -e

if [ "$prune" == "--prune" ]; then
  yes | docker system prune -a
fi

docker build -f bio_check/"$lib"/Dockerfile -t ghcr.io/biosimulators/bio-check-"$lib" ./bio_check/"$lib"

if [ "$run_" == "--run" ]; then
  echo "Running container for $lib"
  ./assets/scripts/run_container.sh "$lib"
fi
