#!/usr/bin/env bash

# script for configuring and installing the complete conda virtual environment from environment.yml at this repo's
# installing Smoldyn. NOTE: The script called below will only work for arm-based Macs. Please refer to the Smoldyn documentation if you are using another platform.


function construct_env {
  arm_platform=$(uname -a | grep "Darwin")
  echo "Creating environment from ./environment.yml on $arm_platform..."
  source /Users/alexanderpatrie/miniconda3/etc/profile.d/conda.sh
  conda run pip3 install --upgrade pip
  conda run pip install --upgrade pip
  conda env create -f ./environment.yml -y
  conda run -n bio-compose-server pip3 install --upgrade pip
  conda run -n bio-compose-server pip install --upgrade pip
  ./assets/dev/scripts/install-smoldyn-mac-silicon.sh || poetry run pip3 install smoldyn
  conda run -n bio-compose-server pip3 install amici biosimulators-amici  # biosimulators-pysces
  # conda install -n bio-compose-server -c conda-forge -c pysces pysces -y
  # poetry config virtualenvs.create false
  # sudo conda run -n bio-compose-server poetry env use 3.10
  # sudo conda run -n bio-compose-server poetry lock
  # sudo conda run -n bio-compose-server poetry install --only=dev
  .
  # poetry run pip install ./api ./worker

  echo "Environment created!"
}

function install_additional_deps {
  conda install -c conda-forge -c pysces pysces -y \
    && conda run pip3 install amici biosimulators-amici biosimulators-pysces
  arm_platform=$(uname -a | grep "Darwin")
  if [ "$arm_platform" != "" ]; then
    ./assets/dev/scripts/install-smoldyn-mac-silicon.sh
  else
    conda run pip install smoldyn
  fi
  pip cache purge
  pip install ./api \
    && pip install ./worker
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

