#!/usr/bin/env bash

cat /app/assets/dropped.txt | xargs -n 1 pip uninstall -y

# Path to the file with the list of dependencies
# DEPS_FILE="/app/assets/dropped.txt"
#
# # Check if the file exists
# if [ ! -f "$DEPS_FILE" ]; then
#   echo "Dependencies file not found!"
#   exit 1
# fi
#
# # Read the file line by line and uninstall each dependency
# while IFS= read -r dep; do
#   echo "Uninstalling $dep..."
#   pipenv uninstall "$dep"
# done < "$DEPS_FILE"
