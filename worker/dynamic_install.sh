#!/usr/bin/env bash

function create_env {
    # create dynamic environment
    conda create -n "$JOB_ID" python=3.10 -y
    conda run -n "$JOB_ID" pip3 install process-bigraph uvicorn typing-extension pyyaml uvicorn pydantic pydantic-settings google-cloud-storage pymongo python-dotenv fastapi requests-toolbelt
}

function install_simulators {
  # install each specified simulator
  for sim in $SIMULATORS; do
    pkg="$sim"
    if [ "$INPUT_FORMAT" == ".omex" ]; then
      pkg=biosimulators-"$pkg"
    fi
    if [ "$sim" == pysces ]; then
      conda install -n "$JOB_ID" -c conda-forge -c pysces pysces
    fi
    conda run -n "$JOB_ID" pip3 install "$pkg"
  done
}


# store registered addresses
while true; do
  INPUT_FORMAT=$(conda run -n server python3 -c "from main import db_connector;print(db_connector.pending_jobs()[0].get('path').split('/')[-1])")
  SIMULATORS=$(conda run -n server python3 -c "from main import db_connector;print(db_connector.pending_jobs()[0].get('simulators'))")
  SIMULATORS=$(echo "$SIMULATORS" | tr -d "[]' ")
  JOB_ID=$(conda run -n server python3 -c "from main import db_connector;print(db_connector.pending_jobs()[0].get('job_id'))")
  IFS=','

  # create the env
  create_env

  # install env simulators as per request
  install_simulators

  # run main once
  conda run -n server python main.py

  # remove env for job
  conda env remove -n "$JOB_ID" -y

  # sleep and run again
  conda run -n server python -c "from time import sleep;print('sleeping...');sleep(5)"

  # TODO: create a handler for no jobs within several minutes
done



