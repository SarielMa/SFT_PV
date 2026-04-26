#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="$SCRIPT_DIR/environment_b200_local.yml"
ENV_NAME="finben_b200"

module purge
module load StdEnv || true
module load CUDA/12.8.0
module load miniconda

if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  # The module sets EBROOTMINICONDA even when `conda` is not yet callable in POSIX shells.
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_BASE="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
  echo "YAML: $ENV_YML"
  echo "Remove it with:"
  echo "  conda env remove -n $ENV_NAME"
  echo "Then recreate with:"
  echo "  bash $0"
  exit 0
fi

echo "Creating conda environment '$ENV_NAME' from $ENV_YML"
conda env create -f "$ENV_YML"

echo
echo "Environment created."
echo "Activate with:"
echo "  conda activate $ENV_NAME"
