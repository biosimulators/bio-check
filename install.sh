#!/usr/bin/env bash

platform="$(python3 -c 'import platform;print(platform.system())')"

# create conda env from env file
conda env create -f environment.yml -y

# install smoldyn if mac
if [ "$platform" == "Darwin" ]; then
  smoldyn_installer=./assets/dev/scripts/install-smoldyn-mac-silicon.sh
  sudo chmod +x "$smoldyn_installer"
  "$smoldyn_installer"
fi

# install deps from pyproject and activate env
conda run -n bio-compose-servser pip install --upgrade pip
conda run -n bio-compose-server pip install -e .
conda activate bio-compose-server