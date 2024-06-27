#!/usr/bin/env bash

function login_prompt {
  echo "Enter your Github User-Name: "
  read -r usr_name
  if docker login ghcr.io -u "$usr_name"; then
    echo "Successfully logged in to GHCR!"
  else
    echo "Could not validate credentials."
    exit 1
  fi
}


usr_name="$1"
cat ~/.ssh/.env | docker login ghcr.io -u "$usr_name" --password-stdin
