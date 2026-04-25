#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="$SCRIPT_DIR/environment_b200_local.yml"
ENV_NAME="finben_b200"

module purge
module load StdEnv || true
module load CUDA/12.8.0
module load miniconda

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
  echo "YAML: $ENV_YML"
  exit 0
fi

echo "Creating conda environment '$ENV_NAME' from $ENV_YML"
conda env create -f "$ENV_YML"

echo
echo "Environment created."
echo "Activate with:"
echo "  conda activate $ENV_NAME"
