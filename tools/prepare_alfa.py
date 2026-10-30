"""ALFA processed CSV release -> data/processed/alfa/{flights/*.parquet, meta.csv}.

Download processed.zip from https://doi.org/10.1184/R1/12707963 and unzip it, then:
  python tools/prepare_alfa.py --src path/to/processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavbench import prep  # noqa: E402
from uavbench.bench import PROC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
a = ap.parse_args()
out = PROC / "alfa"
rows = []
for folder in sorted(p for p in Path(a.src).iterdir() if p.is_dir()):
    if "no_ground_truth" in folder.name:
        continue
    df, info = prep.read_alfa_seq(folder)
    fid = prep.save(df, out, folder.name)
    rows.append(dict(flight_id=fid, group=prep.alfa_group(folder.name), fault_type=prep.alfa_fault(folder.name),
                     src=folder.name, **info))
# sequences cut from one bag or overlapping in time form one group
g = prep.overlap_groups([r["group"] for r in rows], [r["abs_t0"] for r in rows], [r["abs_t1"] for r in rows])
for r, gi in zip(rows, g):
    r["bag"], r["group"] = r["group"], gi
m = prep.write_meta(rows, out)
print(f"{len(m)} sequences, {m.group.nunique()} groups, {int(m.onset_s.notna().sum())} with failures -> {out}")
