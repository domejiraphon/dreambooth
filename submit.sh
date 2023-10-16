#!/bin/bash


SCRIPTS_FOLDER="./$1"
if [ ! -d "$SCRIPTS_FOLDER" ]; then
  echo "Error: Scripts folder not found."
  exit 1
fi

for script in "$SCRIPTS_FOLDER"/*.sh; do
  echo "Running script: $script"
  sbatch "$script"
done

echo "All scripts executed."