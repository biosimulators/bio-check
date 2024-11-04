#!/usr/bin/env bash

set -e

function install_linux_deps {
  # cd ../ (app)
  apt-get update && apt-get install -y build-essential \
      gfortran
      libopenmpi-dev
      libspdlog-dev
      libpugixml-dev
      cmake
      gcc
      g++
      petsc-dev
}

function install_git_repo_deps {
  # clone repos
  git clone https://github.com/FEniCS/ufl.git \
    && git clone -b release https://gitlab.com/petsc/petsc.git petsc

  # install repos into the poetry env
  poetry run pip install --upgrade pip \
    && poetry run --directory . pip install --prefix ./worker ./ufl \
    && poetry run --directory . pip install --prefix ./worker ./petsc
}

function install_poetry_deps {
  poetry add fenics-ffcx fenics-basix nanobind mpi4py \
    && poetry add petsc4py \
    && poetry add scikit-build-core --extras=pyproject
}

function install_dolfinx {
  # clone repo and set up build dir
  git clone https://github.com/FEniCS/dolfinx.git \
    && cd dolfinx/cpp \
    && mkdir build/ \
    && cd build || return

  # build content and return
  cmake ../ \
    && make install \
    && ../assets/install_deps.sh ./dolfinx/cpp/build/build-requirements.txt \
    && poetry run pip install -r build-requirements.txt \
    && poetry run pip install --check-build-dependencies --no-build-isolation . \
    && cd ../../../
}

install_dolfinx