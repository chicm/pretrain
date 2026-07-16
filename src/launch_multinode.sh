#!/usr/bin/env bash
# Compatibility shim. Multi-node orchestration is implemented by the generic
# scripts/launch_multinode.sh and configured by tracked files under recipes/.
set -euo pipefail

echo "DEPRECATED: do not launch an experiment from src/launch_multinode.sh." >&2
echo "Run a tracked script under recipes/ instead." >&2
exit 2
