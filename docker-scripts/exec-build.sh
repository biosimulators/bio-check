#!/bin/zsh

lib="$1"

docker system prune -a -f
cd ./verification_service/$lib

docker build -t spacebearamadeus/verification-service-$lib .
