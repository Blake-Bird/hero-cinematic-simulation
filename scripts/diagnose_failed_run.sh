#!/usr/bin/env bash
# Print the native GADGET error, which Python's CalledProcessError hides.
set -euo pipefail
run="${1:?Usage: bash scripts/diagnose_failed_run.sh /absolute/path/to/run}"
for file in "$run/input/gadget.log" "$run/input/ic_power_spectrum.log" "$run/input/ngenic.log"; do
    if [[ -f "$file" ]]; then
        printf '\n=== %s (last 100 lines) ===\n' "$file"
        tail -n 100 "$file"
    fi
done
printf '\n=== Manifest status ===\n'
if [[ -f "$run/manifest.json" ]]; then
    python3 - "$run/manifest.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k in ('status','gadget_run','prepared_gadget'):
    if k in r:
        print(k, ':', str(r[k])[:2500])
PY
fi
