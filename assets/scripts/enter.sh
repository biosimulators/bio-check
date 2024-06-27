#!/usr/bin/env bash

# Activate the virtual environment
source /app/.venv/bin/activate

# Execute the command passed to the entrypoint script
exec "$@"