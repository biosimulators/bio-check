#!/usr/bin/env bash

requirements_path="$1"

poetry run pip install --upgrade pip
while read -r requirement; do
    poetry add "$requirement"
done < "$requirements_path"
