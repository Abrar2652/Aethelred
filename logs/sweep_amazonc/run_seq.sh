#!/bin/bash
# Amazon-C regularization/architecture diagnosis sweep — FORCE RETRAIN, sequential.
# Isolates the cause of the Aethelred clean-accuracy regression (0.883 -> 0.77).
cd /nas/home/jahin/Aethelred
PY=/usr/bin/python3
TS=$(date +%m%d_%H%M)
OUT=logs/sweep_amazonc
SUM=$OUT/SUMMARY_${TS}.txt
base="--task node --dataset Amazon-C --arch GCN --epochs 200 --force_retrain"

echo "=== Amazon-C sweep (FORCE RETRAIN, sequential) $(date) ===" | tee "$SUM"
echo "baseline NodeGCN (table1_1) = 0.883" | tee -a "$SUM"

run_one () {
  local name="$1"; shift
  local lf="$OUT/${name}_FR_${TS}.log"
  echo "[$(date +%H:%M)] training $name :: ${*:-<defaults>}"
  CUDA_VISIBLE_DEVICES=0 $PY run_aethelred_comparison.py $base "$@" > "$lf" 2>&1
  local acc
  acc=$(grep 'Final test accuracy' "$lf" | grep -oE '[0-9.]+$')
  printf "%-18s = %s\n" "$name" "${acc:-FAILED}" | tee -a "$SUM"
}

run_one c1_default
run_one c2_alloff      --alpha 0 --beta 0 --gamma 0 --delta 0 --epsilon 0
run_one c8_alloff_h20  --alpha 0 --beta 0 --gamma 0 --delta 0 --epsilon 0 --hidden_focal 20
run_one c4_noalpha     --alpha 0
run_one c6_noeps       --epsilon 0

echo "=== SWEEP DONE $(date) ===" | tee -a "$SUM"
