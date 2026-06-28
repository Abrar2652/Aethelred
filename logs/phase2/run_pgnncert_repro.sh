#!/bin/bash
# Reproduce PGNNCert numeric certified-accuracy curves (the SOTA prediction-cert
# curve) on all 8 datasets via the OFFICIAL code. variant=E (edge cert, matches
# our edge-perturbation threat model), GCN backbone, paper-default T, 200 epochs.
# Outputs _ref_pgnncert/results/{graph,node}_<ds>_GCN_T<T>.json with certified
# accuracy at p in [0,1,2,3,5,10,15,20,25,30].
cd /nas/home/jahin/Aethelred/_ref_pgnncert
PY=/usr/bin/python3
LOG=../logs/phase2

launch () {  # gpu kind dataset T
  local gpu=$1 kind=$2 ds=$3 T=$4
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u run_${kind}_experiment.py \
     --dataset "$ds" --method pgnncert --variant E --gnn GCN --T $T --epochs 200 \
     > $LOG/pgnncert_${kind}_${ds}.log 2>&1 &
  echo "  GPU$gpu PGNNCert $kind $ds T=$T PID $!"
}

echo "=== PGNNCert reproduction (GPUs 1,2,3; GPU0 busy w/ XGNNCert-Benzene) ==="
# graph datasets, T=50
launch 1 graph MUTAG    50
launch 2 graph AIDS     50
launch 3 graph PROTEINS 50
launch 1 graph DD       50
# node datasets, T=60
launch 2 node Cora-ML   60
launch 3 node CiteSeer  60
launch 1 node PubMed    60
launch 2 node Amazon-C  60

echo "=== waiting for all PGNNCert jobs ==="
wait
echo "=== PGNNCert reproduction COMPLETE $(date) ==="
ls -lt results/*.json | head -12
