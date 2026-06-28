#!/bin/bash
# Fire all Phase-2 curve jobs with nohup (Phase-1 style; persists across turns).
# Re-run this to relaunch any that node-events killed (each uses force/retrain).
PGN=/nas/home/jahin/Aethelred/_ref_pgnncert
AE=/nas/home/jahin/Aethelred
XG=/nas/home/jahin/Aethelred/baselines/XGNNCert
PY=/usr/bin/python3
LOG=/nas/home/jahin/Aethelred/logs/phase2

pg(){ # gpu kind ds T   -- skip if result already exists
  local out=$PGN/results/$2_$3_GCN_T$4.json
  if [ -f "$out" ]; then echo "  SKIP PGNNCert $2 $3 (done)"; return; fi
  CUDA_VISIBLE_DEVICES=$1 nohup $PY -u $PGN/run_$2_experiment.py --dataset "$3" \
    --method pgnncert --variant E --gnn GCN --T $4 --epochs 200 --retrain \
    > $LOG/pgnncert_$2_$3.log 2>&1 &
  echo "  GPU$1 PGNNCert $2 $3 T$4 PID $!"; }

ae(){ # gpu ds   -- Aethelred cert curves (skip if pred-cert result exists)
  local out=$AE/results/phase2_aethelred_predcert_$2.json
  CUDA_VISIBLE_DEVICES=$1 nohup $PY -u $AE/run_aethelred_certcurves.py --dataset "$2" \
    --epochs 150 --T 50 > $LOG/aethelred_cc_$2.log 2>&1 &
  echo "  GPU$1 Aethelred-certcurves $2 PID $!"; }

cd $PGN
echo "== PGNNCert SOTA (graph T50, node T60) =="
pg 0 graph MUTAG 50;  pg 1 graph AIDS 50;  pg 2 graph PROTEINS 50;  pg 3 graph DD 50
pg 0 node PubMed 60;  pg 1 node Amazon-C 60
echo "== Aethelred overlay cert-curves (graph) =="
ae 2 MUTAG; ae 3 AIDS; ae 0 PROTEINS; ae 1 DD
echo "== XGNNCert explanation curve (Benzene) =="
CUDA_VISIBLE_DEVICES=2 nohup $PY -u $XG/run_xgnncert_baseline.py --dataset Benzene \
  --T 70 --p 0.3 --epochs 80 --k 12 --tau 0.3 --max_test 30 --budgets 0 1 2 3 4 6 8 10 \
  > $LOG/xgnncert_Benzene_real.log 2>&1 &
echo "  GPU2 XGNNCert Benzene PID $!"
echo "launched $(date)"
