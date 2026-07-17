#!/usr/bin/env bash
# Emergency multi-node training-process cleanup.
# NODES is a whitespace-separated host list; TRUST_USER owns inter-node SSH.
set -euo pipefail

TRUST_USER=${TRUST_USER:?set TRUST_USER}
NODES=${NODES:?set NODES to a whitespace-separated host list}

for node in $NODES; do
  su "$TRUST_USER" -c "ssh -o StrictHostKeyChecking=no $node 'rocm-smi --showpids 2>/dev/null | grep -oE \"^[0-9]+\" | xargs -r kill -9 2>/dev/null; pgrep -x torchrun | xargs -r kill -9 2>/dev/null; true'" 2>/dev/null &
done
wait
echo "kill signal sent to configured nodes"
