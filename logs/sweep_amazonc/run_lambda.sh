#!/bin/bash
# Amazon-C gate_lambda sweep — validates the residual-gating fix.
# Default losses ON, hidden=64 (node default). Only gate_lambda varies.
# Expect: lambda=1.0 reproduces ~0.772 (regression); lambda->0 recovers ~baseline 0.883.
cd /nas/home/jahin/Aethelred
PY=/usr/bin/python3
TS=$(date +%m%d_%H%M)
OUT=logs/sweep_amazonc
SUM=$OUT/LAMBDA_${TS}.txt
base="--task node --dataset Amazon-C --arch GCN --epochs 200 --force_retrain"

echo "=== Amazon-C gate_lambda sweep $(date) ===" | tee "$SUM"
echo "baseline NodeGCN (table1_1, ungated) = 0.883" | tee -a "$SUM"

for gl in 1.0 0.5 0.3 0.1 0.0; do
  lf="$OUT/gl${gl}_${TS}.log"
  echo "[$(date +%H:%M)] training gate_lambda=$gl"
  CUDA_VISIBLE_DEVICES=0 $PY run_aethelred_comparison.py $base --gate_lambda $gl > "$lf" 2>&1
  acc=$(grep 'Final test accuracy' "$lf" | grep -oE '[0-9.]+$')
  printf "gate_lambda=%-5s = %s\n" "$gl" "${acc:-FAILED}" | tee -a "$SUM"
done
echo "=== LAMBDA SWEEP DONE $(date) ===" | tee -a "$SUM"
