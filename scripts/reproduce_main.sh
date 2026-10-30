#!/usr/bin/env bash
# Main results (Table 3): ALFA at 100% labels, RflyMAD-Real and UAV-SEAD at 5/20/100%, plus the log-level split.
set -eu
cd "$(dirname "$0")/.."
OUT=${OUT:-results/records}
python tools/run_bench.py --ds alfa --chset F --budgets 1.0 --out $OUT
python tools/run_bench.py --ds rflymad --chset F --splits 0 1 2 3 4 --budgets 0.05 0.2 1.0 --out $OUT
python tools/run_bench.py --ds uavsead --chset F --splits 0 1 2 --budgets 0.05 0.2 1.0 --out $OUT
python tools/run_bench.py --ds uavsead --chset F --splits 0 1 2 --budgets 1.0 --uavsead-group log --tag _grouplog --out $OUT
