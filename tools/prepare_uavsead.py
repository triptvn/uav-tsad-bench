"""UAV-SEAD -> data/processed/uavsead/{flights/*.parquet, meta.csv}.

Logs are fetched one by one from Hugging Face (aykutkabaoglu/uav-flight-anomaly-dataset, about 12 GB for the
annotated subset) and cached in --raw. Logs annotated only as "Uncategorized" are skipped.
  python tools/prepare_uavsead.py --raw data/uavsead_raw --workers 8
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavbench import prep  # noqa: E402
from uavbench.bench import PROC  # noqa: E402

BASE = "https://huggingface.co/datasets/aykutkabaoglu/uav-flight-anomaly-dataset/resolve/main/"
OUT = PROC / "uavsead"


def job(key, anns, raw):
    safe = key.replace("/", "__")
    classes = sorted({x["class"] for x in anns} - {"Normal"}) or ["Normal"]
    row = dict(flight_id=safe, fault_type="+".join(classes))
    f = Path(raw) / (safe + ".ulg")
    if not f.exists():
        tmp = f.with_suffix(".part")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(BASE + "ulg_files/" + key + ".ulg", tmp)
                tmp.rename(f)
                break
            except Exception as e:
                if attempt == 2:
                    return dict(row, error=f"download: {e!r}")
                time.sleep(5)
    try:
        df, info = prep.read_uavsead_log(f, anns)
    except Exception as e:
        return dict(row, error=f"parse: {e!r}")
    prep.save(df, OUT, key)
    return dict(row, **info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/uavsead_raw")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    Path(a.raw).mkdir(parents=True, exist_ok=True)
    mp = Path(a.raw) / "mapping.json"
    if not mp.exists():
        urllib.request.urlretrieve(BASE + "mapping.json", mp)
    m = json.load(open(mp))
    todo = {k: v["annotations"] for k, v in m.items() if {x["class"] for x in v["annotations"]} != {"Uncategorized"}}
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, k, v, a.raw) for k, v in sorted(todo.items())]
        for i, fu in enumerate(as_completed(futs)):
            rows.append(fu.result())
            if i % 100 == 0:
                print(i, "/", len(futs), flush=True)
    meta = prep.write_meta(rows, OUT)
    print(f"{len(meta)} logs, errors {int(meta['error'].notna().sum()) if 'error' in meta else 0} -> {OUT}")
