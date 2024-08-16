#!/usr/bin/env bash

set -e

version="$1"
un=AlexPatrie

./assets/scripts/build_api.sh "$version" "$un" && ./assets/scripts/build_worker.sh "$version" "$un"

