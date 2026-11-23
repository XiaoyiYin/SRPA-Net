#!/usr/bin/env bash
# Start (or resume) the selected single-variant run only when explicitly invoked.
set -euo pipefail
srpa_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
srpa_run_dir="${1:-/data/SRPA-Net/runs/20261016_142306_full224_selected_spectral_detail}"
exec /home/chic/miniconda3/envs/SpecAware/bin/python -u \
  "$srpa_root/run_experiments.py" \
  --config "$srpa_root/configs/selected_spectral_detail.json" \
  --run-dir "$srpa_run_dir"
