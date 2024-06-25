#!/bin/zsh

lib="$1"

if [ "${lib}" == "" ]; then
  echo "You must enter one of: api, storage, worker library names as a runtime arg."
  exit 1
fi

docker system prune -a -f
cd ../verification_service/$lib

docker build -t spacebearamadeus/verification-service-$lib .
