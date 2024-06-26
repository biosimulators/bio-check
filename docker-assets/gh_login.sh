#!/usr/bin/env bash

echo "Enter your Github User-Name: "
read -r usr_name

if docker login ghcr.io -u "$usr_name"; then
  echo "Successfully logged in to GHCR!"
else
  echo "Could not validate credentials."
  exit 1
fi