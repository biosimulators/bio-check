#!/usr/bin/env bash

python3 -c "from main import db_connector as conn;conn.refresh_jobs();print('Jobs refreshed!')"
