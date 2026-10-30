#!/usr/bin/env bash
# Ablation (Table 4) on RflyMAD-Real and ALFA.
set -eu
cd "$(dirname "$0")/.."
OUT=${OUT:-results/records}
UNSUP="PCA IForest HBOS USAD FITS"
WIN="MovingVar SqDiff PCA IForest HBOS USAD FITS LightGBM ExtraTrees CNN TCN LSTM"
python tools/run_bench.py --ds alfa --chset FA --budgets 1.0 --out $OUT
python tools/run_bench.py --ds rflymad --methods $UNSUP --budgets 0.05 0.2 1.0 --unsup-normal --tag _normal --out $OUT
python tools/run_bench.py --ds alfa --methods $UNSUP --budgets 1.0 --unsup-normal --tag _normal --out $OUT
python tools/run_bench.py --ds rflymad --chset K --budgets 0.05 1.0 --out $OUT
python tools/run_bench.py --ds rflymad --methods $WIN --budgets 0.05 1.0 --window 16 --tag _w16 --out $OUT
python tools/run_bench.py --ds rflymad --methods $WIN --budgets 0.05 1.0 --window 64 --tag _w64 --out $OUT
python tools/run_bench.py --ds rflymad --methods LightGBM TCN --splits 0 1 --budgets 1.0 --hz 50 --window 160 --tag _hz50 --out $OUT
