#!/usr/bin/env bash

# Args:
# $1: the library for which you would like to build a container
# -p ($2): whether to prune docker system before building anything.
# -b ($3): whether to build the base image before building library
# -r ($4): whether to run the library container after build
# Run at root of repo!

lib="$1"
prune="$2"  # -p
build_="$3"  # -b
run_="$4"

if [ "$prune" ]; then
  docker system prune -a -f
fi

if [ "$build_" ]; then
  ./build_base.sh
fi

cd bio_check/"$lib" || exit
docker build -t ghcr.io/biosimulators/bio-check-"$lib" ./bio_check/"$lib"

if [ "$run_" ]; then
  echo "Running container for $lib"
  ./run_container.sh "$lib"
fi
