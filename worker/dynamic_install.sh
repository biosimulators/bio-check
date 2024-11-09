#!/usr/bin/env bash

# num_simulators=$(poetry run python3 -c "print(len($simulators))")

while true; do
  INPUT_FORMAT=$(poetry run python3 -c "from main import db_connector;print(db_connector.pending_jobs()[0].get('path').split('/')[-1]")
  SIMULATORS=$(poetry run python3 -c "from main import db_connector;print(db_connector.pending_jobs()[0].get('simulators'))")
  SIMULATORS=$(echo "$SIMULATORS" | tr -d "[]' ")
  JOB_ID=$(poetry run python3 -c "from main import db_connector;print(db_connector.pending_jobs()[0].get('job_id'))")
  IFS=','

  # create dynamic environment
  conda create -n "$JOB_ID" python=3.10 -y

  # install each specified simulator
  for sim in $SIMULATORS; do
    pkg="$sim"
    if [ "$INPUT_FORMAT" == ".omex" ]; then
      pkg=biosimulators-"$pkg"
    fi

    if [ "$sim" == "pysces" ]; then
      conda install -n "$JOB_ID" -c conda-forge -c pysces pysces
    fi

    conda run -n "$JOB_ID" pip3 install "$pkg"
  done

  # TODO:
  # 1. in base, upgrade base conda pip but do NOT create env
  # 2. in api, install BASE requirements(mostly api anyway!)
  # 3. in worker run script, get simulators and req input format and create env indexed by job_id from BASE requirements
  # 4. For each requested simulator, dynamically install into existing base
  # 5. Run main.py but instead of while loop, let it run just for the single job.
  # 6. Db is already updated with results after number 5, so run conda env remove -n "$JOB_ID" -y
done



