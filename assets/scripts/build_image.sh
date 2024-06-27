#!/usr/bin/env bash

# Args:
# $1: the library for which you would like to build a container
# --run: whether to run the library container after build
# --prune: whether to prune docker system before building anything.
# Run at the root of the repo!

set -e

# Default values for flags
run_=false
prune=false

# Parse arguments
lib=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      run_=true
      shift # Remove --run from processing
      ;;
    --prune)
      prune=true
      shift # Remove --prune from processing
      ;;
    *)
      if [[ -z "$lib" ]]; then
        lib="$1"
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      shift # Remove the argument from processing
      ;;
  esac
done

if [[ -z "$lib" ]]; then
  echo "You must specify the library to build."
  exit 1
fi

# Prune the Docker system if --prune flag is set
if [ "$prune" = true ]; then
  yes | docker system prune -a
fi

# Build the Docker image
docker build -f bio_check/"$lib"/Dockerfile -t ghcr.io/biosimulators/bio-check-"$lib" ./bio_check/"$lib"

# Run the container if --run flag is set
if [ "$run_" = true ]; then
  echo "Running container for $lib"
  ./assets/scripts/run_container.sh "$lib"
else
  echo "Build complete. Not running the container."
fi
