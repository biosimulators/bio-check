#!/bin/zsh

lib="$1"

if [ "${lib}" == "" ]; then
  echo "You must enter one of: api, storage, worker library names as a runtime arg."
  exit 1
fi

docker run -d -p 8001:3001 spacebearamadeus/verification-service-$lib
