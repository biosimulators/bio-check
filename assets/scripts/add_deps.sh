#!/usr/bin/env bash

packages="$1"

echo "$packages" > .req

req=req=$(cat .req)

poetry add "$req" \
    && rm .req