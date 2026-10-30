#!/usr/bin/env bash
# Measurement queue, run ON the Raspberry Pi next to edge_bench.py and pkg/. Finished results are skipped.
# One process per (dataset, detector, threads) so that the resident memory is measured per detector.
# Waits until the SoC is below 65 C before each run and logs temperature and throttling flags every 10 s.
set -u
cd "$(dirname "$0")"
exec 9>run.lock
flock -n 9 || { echo "already running"; exit 0; }
mkdir -p results logs
( while true; do echo "$(date +%s),$(vcgencmd measure_temp | grep -oE '[0-9.]+'),$(vcgencmd get_throttled | cut -d= -f2)" >> logs/thermal.csv; sleep 10; done ) &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT
temp() { vcgencmd measure_temp | grep -oE '[0-9]+' | head -1; }
for ds in rflymad uavsead alfa; do
  models=$(venv/bin/python -c "import json; print(' '.join(m['method'] for m in json.load(open('pkg/$ds/meta.json'))['models']))")
  for m in $models; do
    for t in 1 4; do
      out=results/${ds}_${m}_t${t}.json
      [ -f "$out" ] && continue
      while [ "$(temp)" -ge 65 ]; do sleep 10; done
      t0=$(vcgencmd get_throttled | cut -d= -f2)
      if venv/bin/python edge_bench.py run --pkg pkg/$ds --threads $t --only $m --steps 2000 --out "$out.tmp" \
           > logs/${ds}_${m}_t${t}.log 2>&1; then
        t1=$(vcgencmd get_throttled | cut -d= -f2)
        venv/bin/python - "$out.tmp" "$t0" "$t1" <<'EOF'
import json, sys
p, a, b = sys.argv[1:]
d = json.load(open(p)); d["throttled_before"], d["throttled_after"] = a, b
json.dump(d, open(p, "w"), indent=1)
EOF
        mv "$out.tmp" "$out"
      else
        echo "$(date '+%F %T') failed $ds $m t$t" >> logs/errors.txt
      fi
    done
  done
done
echo "$(date '+%F %T') done" >> logs/done.txt
