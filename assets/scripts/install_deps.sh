#!/usr/bin/env bash

requirements_path="$1"

while read -r requirement; do
    poetry add "$requirement"
done < "$requirements_path"
