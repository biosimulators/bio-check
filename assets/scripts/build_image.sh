#!/usr/bin/env bash

# Args:
# $1: the library for which you would like to build a container
# --run: whether to run the library container after build
# --prune: whether to prune docker system before building anything.
# Run at the root of the repo!

set -e

# Default values for flags
# run_=false
# prune=false
#
# # Parse arguments
# lib=""
#
# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     --run)
#       run_=true
#       shift # Remove --run from processing
#       ;;
#     --prune)
#       prune=true
#       shift # Remove --prune from processing
#       ;;
#     *)
#       if [[ -z "$lib" ]]; then
#         lib="$1"
#       else
#         echo "Unknown argument: $1"
#         exit 1
#       fi
#       shift # Remove the argument from processing
#       ;;
#   esac
# done

lib="$1"
version="$2"
prune="$3"

if [[ -z "$lib" ]]; then
  echo "You must specify the library to build."
  exit 1
fi

# Prune the Docker system if --prune flag is set
if [ "$prune" == "--prune" ]; then
  yes | docker system prune -a
fi

# Build the Docker image
docker build --platform linux/amd64 -f ./"$lib"/Dockerfile-"$lib" -t ghcr.io/biosimulators/bio-check-"$lib":"$version" ./"$lib"

# kubectl create secret generic gcp-credentials --from-file=key.json=/Users/alexanderpatrie/.ssh/bio-check-428516-eb623914aa25.json --namespace=dev --dry-run=client -o yaml > gcp-secret.yaml
