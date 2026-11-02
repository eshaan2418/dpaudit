#!/bin/bash
# Main grid: 6 privacy levels x 3 seeds, 10 epochs, 2 parallel workers x 4 threads.
cd "$(dirname "$0")/.."
for e in 0.5 1 2 4 8 inf; do for s in 0 1 2; do echo "--eps $e --seed $s"; done; done | \
  xargs -P 2 -I{} sh -c 'gtimeout 2700 nice -n 10 .venv/bin/python scripts/experiment.py {} 2>/dev/null || echo "FAILED {}"'
echo ALLDONE
