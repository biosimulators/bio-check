#!/usr/bin/env bash

image="$1"  # which lib container to run (base, gateway, compose_worker)
version="$2"
use_biocompose="$3"

if [ "$use_biocompose" == "" ]; then
  use_biocompose=1
fi

if [ "$version" == "" ]; then
  if [ "$image" == "base" ]; then
    version=latest
  else
    version=$(cat "$image/.VERSION")
  fi
fi

if [ "$use_biocompose" == 1 ]; then
  img="compose-server-$image"
else
  img="$image"
fi
# docker run --platform linux/amd64 -it -p 8000:3001 ghcr.io/biosimulators/bio-check-"$lib"

docker run --name "$img" --platform linux/amd64 --entrypoint /usr/bin/env -it "$img":"$version" bash
