#!/usr/bin/env bash
# Compatibility shim. Per-node torchrun commands are now assembled from a
# tracked experiment recipe by scripts/launch_multinode.sh.
set -euo pipefail

echo "DEPRECATED: do not launch an experiment from src/run_multinode.sh." >&2
echo "Run a tracked script under recipes/ instead." >&2
exit 2
