#!/usr/bin/env bash

api_version="$1"
worker_version="$2"


echo "Build and Push API Gateway"
./assets/scripts/build_image.sh api "$api_version" && ./assets/scripts/push_image.sh api "$api_version" AlexPatrie

echo "Build and Push Worker"
./assets/scripts/build_image.sh worker "$worker_version" && ./assets/scripts/push_image.sh worker "$worker_version" AlexPatrie

echo "Commit Changes"
./commit.sh
