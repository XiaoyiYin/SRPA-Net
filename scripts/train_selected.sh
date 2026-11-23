#!/usr/bin/env bash
# Start (or resume) the main SRPA-Net experiment.
set -euo pipefail
srpa_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
srpa_run_dir="${1:-$srpa_root/runs/srpa_net}"
exec python -u \
  "$srpa_root/run_experiments.py" \
  --config "$srpa_root/configs/selected_spectral_detail.json" \
  --run-dir "$srpa_run_dir"
