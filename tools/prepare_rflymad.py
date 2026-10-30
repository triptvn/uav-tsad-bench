"""RflyMAD real flights -> data/processed/rflymad/{flights/*.parquet, meta.csv} (stored at 50 Hz).

Point --src at a copy of the RflyMAD Kaggle dataset (xianglile/rflymad) that contains Real-Motor, Real-Sensors
and Real-No_Fault; the script reads the .ulg files and the TestInfo sheets. About 10 GB of input.
  python tools/prepare_rflymad.py --src path/to/rflymad --workers 4
"""
import argparse
import os
import sys
import traceback
from glob import glob
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavbench import prep  # noqa: E402
from uavbench.bench import PROC  # noqa: E402

OUT = PROC / "rflymad"


def job(u):
    parts = ("Real-" + u.split("/Real-", 1)[1]).split("/")
    folder, status = parts[0], parts[1] if len(parts) > 3 else ""
    fid = "/".join(parts[:-1])
    try:
        df, info = prep.read_rflymad_case(u, hz=50.0)
        return dict(flight_id=prep.save(df, OUT, fid), folder=folder, status=status, **info)
    except Exception:
        return dict(flight_id=fid, folder=folder, status=status, error=traceback.format_exc()[-800:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    roots = [p for p in glob(a.src + "/**/Real-*", recursive=True) if os.path.isdir(p)]
    ulgs = sorted({u for r in roots for u in glob(r + "/**/*.ulg", recursive=True)})
    with Pool(a.workers) as pool:
        rows = pool.map(job, ulgs, chunksize=2)
    m = prep.write_meta(rows, OUT)
    print(f"{len(m)} logs, errors {int(m['error'].notna().sum()) if 'error' in m else 0} -> {OUT}")
