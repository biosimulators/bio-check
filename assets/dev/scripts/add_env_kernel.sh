#!/usr/bin/env bash

env="$1"

if [ "$env" == "" ]; then
  env=$(conda env list | grep '*' | awk '{print $1}')
fi

# install ipykernel if needed
kernel=$(conda list | grep ipykernel)

if [ "$kernel" == "" ]; then
  conda run -n "$env" pip install ipykernel jupyterlab
fi

python_version=$(conda run -n "$env" python --version)
conda run -n "$env" python -m ipykernel install --user --name="$env" --display-name "BioCompose Server($env): $python_version"
