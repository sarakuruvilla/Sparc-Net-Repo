#!/usr/bin/env bash
# Full revision grid. Each call is independent and skips finished runs, so the grid can be
# split across machines / Colab sessions and resumed safely.
#   12 stations x 4 horizons x (3 reference + 4 neural x SEEDS) runs.
# Rough CPU cost: ~3-6 min per neural run on the Beijing data -> use a GPU or run in parallel
# (e.g. one station per process:  STATIONS="Dongsi" bash run_all.sh).
set -e
cd "$(dirname "$0")"
STATIONS=${STATIONS:-"Aotizhongxin Changping Dingling Dongsi Guanyuan Gucheng Huairou Nongzhanguan Shunyi Tiantan Wanliu Wanshouxigong"}
HORIZONS=${HORIZONS:-"1 6 12 24"}
SEEDS=${SEEDS:-"7 17 27 37 47"}
for st in $STATIONS; do
  for h in $HORIZONS; do
    for m in persistence seasonal arfima; do python run_real.py --station $st --horizon $h --model $m; done
    for s in $SEEDS; do
      for m in gru arfima_gru spatial sparc; do python run_real.py --station $st --horizon $h --model $m --seed $s; done
    done
  done
done
python dm_test.py
# synthetic testbed, corrected and legacy configuration, all coupling levels
for c in 0.0 0.5 1.0 2.0 4.0; do
  for s in $SEEDS; do
    python run_synthetic_fixed.py --seed $s --coupling $c
    python run_synthetic_fixed.py --seed $s --coupling $c --legacy
  done
done
