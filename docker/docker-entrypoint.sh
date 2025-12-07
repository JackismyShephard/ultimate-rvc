#!/bin/bash
# This script runs every time the container starts.
# 'set -e' ensures that the script will exit immediately if any command fails.
set -e

# Define the path to the models directory inside the container.
# This is the path where your docker-compose.yml will mount the host volume.
MODELS_DIR="/app/models"

# Check if a key subdirectory exists. If not, we assume this is the first run
# and the models directory needs to be initialized.
if [ ! -d "${MODELS_DIR}/rvc" ]; then
  echo "--------------------------------------------------------------------------"
  echo "INFO: Models directory appears to be uninitialized."
  echo "INFO: Running one-time model download. This may take a while..."
  echo "--------------------------------------------------------------------------"
  
  # Run the application's own initialization script to download all models.
  # We must run this command from within the virtual environment.
  ./uv/.venv/bin/python ./src/ultimate_rvc/core/main.py
  
  echo "--------------------------------------------------------------------------"
  echo "INFO: Model download complete. Starting web interface..."
  echo "--------------------------------------------------------------------------"
else
  echo "INFO: Models directory already exists. Skipping download and starting web interface."
fi

# This is a crucial line. `exec "$@"` replaces the script process with the
# command passed as arguments to the script. In our case, this will be the
# CMD from the Dockerfile: ["./urvc", "run", "--listen", "--listen-host", "0.0.0.0"]
# This ensures the web server runs as the main container process.
exec "$@"