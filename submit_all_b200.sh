#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

scripts=(
  "apply_server_ti_dpo_b200.sh"
  "apply_server_tdpo_b200.sh"
  "apply_server_ipo_b200.sh"
  "apply_server_dpop_b200.sh"
  "apply_server_dpo_b200.sh"
  "apply_server_cal_dpo_b200.sh"
)

for script in "${scripts[@]}"; do
  echo "Submitting $script"
  sbatch "$script"
done
