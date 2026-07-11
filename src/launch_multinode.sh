#!/bin/bash
# Launcher: run from node-0. Fans out run_multinode.sh to all nodes via
# `su $TRUST_USER -c "ssh node-N ..."`. Node-0 is master.
# TRUST_USER = the inter-node SSH trust user. CONDA_ENV = the torch conda env.
# WORKDIR = shared work dir (code synced to local disk per node in practice).
set -e
NODES=(node-0 node-1 node-2 node-3)
MASTER_ADDR=$(getent hosts node-0 | awk '{print $1}')
export MASTER_ADDR
WORKDIR=${WORKDIR:?set WORKDIR}
TRUST_USER=${TRUST_USER:?set TRUST_USER}
CONDA_ENV=${CONDA_ENV:-base}

for i in "${!NODES[@]}"; do
  n=${NODES[$i]}
  echo "launching NODE_RANK=$i on $n (master=$MASTER_ADDR)"
  su "$TRUST_USER" -c "ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no $n \
    'cd $WORKDIR/src && source ~/.bashrc && conda activate $CONDA_ENV 2>/dev/null; \
     NNODES=4 GPUS_PER_NODE=8 NODE_RANK=$i MASTER_ADDR=$MASTER_ADDR MASTER_PORT=29500 \
     nohup bash run_multinode.sh > $WORKDIR/train_node${i}.log 2>&1 &'" &
done
wait
echo "all nodes launched. tail logs at $WORKDIR/train_node*.log"
