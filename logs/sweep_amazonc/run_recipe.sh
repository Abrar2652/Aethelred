#!/bin/bash
# Amazon-C training-recipe diagnosis. Gating OFF everywhere (gate_lambda=0, ruled out).
# Vary lr (baseline=0.002 vs Aethelred-default=0.0005) and num_envs (1=clean only vs 5).
# Baseline NodeGCN recipe = lr 0.002, single clean graph, ungated -> 0.883.
cd /nas/home/jahin/Aethelred
PY=/usr/bin/python3
TS=$(date +%m%d_%H%M)
OUT=logs/sweep_amazonc
SUM=$OUT/RECIPE_${TS}.txt
base="--task node --dataset Amazon-C --arch GCN --epochs 200 --force_retrain"

echo "=== Amazon-C training-recipe sweep $(date) ===" | tee "$SUM"
echo "baseline NodeGCN (lr0.002, env1, ungated) = 0.883" | tee -a "$SUM"

run_one () {  # name  <extra args>
  local name="$1"; shift
  local lf="$OUT/${name}_${TS}.log"
  echo "[$(date +%H:%M)] $name :: $*"
  CUDA_VISIBLE_DEVICES=0 $PY run_aethelred_comparison.py $base "$@" > "$lf" 2>&1
  local acc; acc=$(grep 'Final test accuracy' "$lf" | grep -oE '[0-9.]+$')
  printf "%-22s = %s\n" "$name" "${acc:-FAILED}" | tee -a "$SUM"
}

# baseline-matched recipe (lr0.002, clean only, ungated): does it recover 0.88?
run_one r_lr002_env1_gl0  --gate_lambda 0 --num_envs 1 --lr 0.002
# isolate env effect (baseline lr, but 5 edge-dropped envs)
run_one r_lr002_env5_gl0  --gate_lambda 0 --num_envs 5 --lr 0.002
# isolate lr effect (clean only, low Aethelred-default lr)
run_one r_lr0005_env1_gl0 --gate_lambda 0 --num_envs 1 --lr 0.0005
# gating effect at the GOOD recipe (lr0.002, clean only, FULL gating)
run_one r_lr002_env1_gl1  --gate_lambda 1 --num_envs 1 --lr 0.002

echo "=== RECIPE SWEEP DONE $(date) ===" | tee -a "$SUM"
