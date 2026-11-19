#!/usr/bin/env bash
# Full pipeline, from the raw StatsBomb files to tables, figures and the interactive page.
# Usage: ./run_all.sh [statsbomb-open-data checkout]   (run ./fetch_data.sh first). About 10 minutes on 2 cores.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="${1:-$HERE/statsbomb-open-data}"
PY="${PYTHON:-python3}"
cd "$HERE"
"$PY" src/verify_data.py "$SB"
"$PY" src/build_events.py "$SB/data"                      # chances, goals, dead time, players on the pitch
(cd src && "$PY" segments.py)                             # time segments for every event type
(cd src && "$PY" analysis.py | tee ../results/analysis.log)          # pre-registered analyses
(cd src && "$PY" exploratory.py | tee ../results/exploratory.log)    # exploratory analyses (labelled)
(cd src && "$PY" summaries.py)                            # effects in extra chances
(cd src && "$PY" figures.py)
"$PY" explorer/export.py
"$PY" explorer/build.py
"$PY" -m pytest -q
