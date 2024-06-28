#!/usr/bin/env bash

cat /app/assets/dropped.txt | xargs -n 1 pip uninstall -y