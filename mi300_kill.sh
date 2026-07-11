#!/bin/bash
# Kill old training on all nodes via GPU compute pids (no self-match).
# TRUST_USER = the inter-node SSH trust user (required).
TRUST_USER=${TRUST_USER:?set TRUST_USER}
for n in node-0 node-1 node-2 node-3; do
  su "$TRUST_USER" -c "ssh -o StrictHostKeyChecking=no $n 'rocm-smi --showpids 2>/dev/null | grep -oE \"^[0-9]+\" | xargs -r kill -9 2>/dev/null; pgrep -x torchrun | xargs -r kill -9 2>/dev/null; true'" 2>/dev/null &
done
wait
echo "kill signal sent to all nodes"
