#!/usr/bin/env bash

env=$(conda env list | grep '*' | awk '{print $1}')

# add kernel
./assets/dev/scripts/add_env_kernel.sh "$env"

# start jupyterlab
conda run -n "$env" jupyter lab
