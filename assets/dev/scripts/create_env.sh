#!/usr/bin/env bash

# script for configuring and installing the complete conda virtual environment from environment.yml at this repo's
# installing Smoldyn. NOTE: The script called below will only work for arm-based Macs. Please refer to the Smoldyn documentation if you are using another platform.


function construct_env {
  echo "Creating environment from ./environment.yml..."
  source /Users/alexanderpatrie/miniconda3/etc/profile.d/conda.sh
  conda clean --all -y \
    && conda env create -f ./environment.yml -y \
    && conda activate bio-compose-server-dev
  echo "Environment created!"
}

function install_deps {
  conda run pip3 install --upgrade pip \
    && conda run pip install --upgrade pip \
    && ./assets/dev/scripts/install-smoldyn-mac-silicon.sh \
    && conda install -c conda-forge -c pysces pysces -y \
    && conda run pip3 install biosimulators-amici
}

function create_env {
  if construct_env; then
    install_additional_deps
  else
    echo "Could not create env. Exiting..."
    exit 1
  fi
}

create_env

