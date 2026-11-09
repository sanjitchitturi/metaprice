#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py --config configs/default.yaml --data_path "${1:-data/online_retail_II.xlsx}" --mode train
