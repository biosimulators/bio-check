#!/usr/bin/env bash

lib="$1"  # which lib container to run
docker run --platform linux/amd64 -it -p 8000:3001 ghcr.io/biosimulators/bio-check-"$lib"
