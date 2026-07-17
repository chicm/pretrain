#!/usr/bin/env bash
# Backward-compatible entry point.
#
# Cluster paths and experiment arguments no longer live in this generic file.
# Run a tracked experiment recipe instead, for example:
#   bash recipes/chimera_8b_1t.sh
set -euo pipefail

echo "DEPRECATED: mi300_mn.sh no longer contains an experiment configuration." >&2
echo "Run a tracked script under recipes/ instead." >&2
exit 2
