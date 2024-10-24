#!/usr/bin/env bash

arg1="$1"  # base version
arg2="$2"  # prune
arg3="$3"  # deploy base after building
arg4="$4"  # run orchestration after docker compose (up)

function prune_step {
  if [ "$arg2" == "-p" ]; then
    docker system prune -a -f
    yes | docker buildx prune --all
  fi
}

function base_step {
  ./assets/scripts/build_base.sh "$arg1"
  if [ "$arg3" == "-d" ]; then
    ./assets/scripts/push_base.sh "$arg1"
  fi
}

function services_step {
  if [ "$arg4" == "-r" ]; then
  ./assets/scripts/build_all.sh "$arg4"
  else
    ./assets/scripts/build_all.sh
  fi
}

function run_pipeline {
  prune_step
  base_step
  services_step
}

run_pipeline


