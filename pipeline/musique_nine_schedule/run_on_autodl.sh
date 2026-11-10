#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: bash run_on_autodl.sh EXISTING_MUSIQUE_STUDY_ROOT ORIGINAL_BGE_MODEL_DIRECTORY NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
if [ "$(uname -s)" != "Linux" ]; then
  echo "Run on the authorized AutoDL Linux host; local Mac execution is forbidden." >&2
  exit 2
fi
package_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python3 "$package_dir/run_multiround.py" --study-root "$1" --model "$2" --out "$3" --preflight-only
python3 "$package_dir/run_multiround.py" --study-root "$1" --model "$2" --out "$3" --execute
