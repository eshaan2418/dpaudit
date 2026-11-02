#!/bin/bash
cd ~/thirty/dpaudit
{ for lr in 0.125 0.25 0.5 1.0 2.0 4.0; do echo "--eps 2 --lr $lr --epochs 3 --tune"; done
  for lr in 0.02 0.05 0.1 0.2; do echo "--eps inf --lr $lr --epochs 3 --tune"; done; } | \
  xargs -P 2 -I{} sh -c 'gtimeout 2700 nice -n 10 .venv/bin/python scripts/experiment.py {} 2>/dev/null || echo "FAILED {}"'
echo TUNEDONE
