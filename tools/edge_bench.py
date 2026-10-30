"""Streamed per-step latency (normalisation + buffering + feature extraction + inference) and memory.

Each step receives one sample, updates a ring buffer and scores that step with batch size 1.
The package is built with `export` on the training machine and copied to the device.
Usage:
  local:  myenv/bin/python src/tools/edge_bench.py export --ds rflymad --out edge/pkg
  Pi:     python3 edge_bench.py run --pkg pkg --threads 1 4 --steps 2000 --out results.json
"""
import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

W, CTX, H = 32, 512, 4


def win_features(w):
    """Single-window version of bench.win_features: (W, C) -> (8C,), same feature order."""
    d = np.abs(np.diff(w, axis=0))
    return np.concatenate([w.mean(0), w.std(0), w.min(0), w.max(0), w[-1], w[-1] - w[0], d.mean(0), d.max(0)])


class Stream:
    def __init__(self, kind, path, threads, fnorm=None):
        self.kind, self.path = kind, Path(path)
        n = np.load(self.path.parent / "norm.npz")
        self.mu, self.sd, self.keep = n["mu"], n["sd"], n["keep"]
        self.C = len(self.keep)
        L = CTX if kind == "tspulse" else W
        self.buf = np.zeros((L, self.C), np.float32)
        self.t = 0
        if self.path.suffix == ".onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        elif self.path.suffix == ".joblib":
            import joblib
            self.m = joblib.load(self.path)
            if hasattr(self.m, "n_jobs"):
                self.m.n_jobs = threads
            if hasattr(self.m, "set_params") and "n_jobs" in self.m.get_params():
                self.m.set_params(n_jobs=threads)
        elif self.path.suffix == ".npz":
            self.p = dict(np.load(self.path, allow_pickle=True))
        elif self.path.suffix == ".pt":
            import torch
            torch.set_num_threads(threads)
            self.m = torch.jit.load(str(self.path)).eval()
        self.fmu = self.fsd = None
        if fnorm:  # feature standardisation of PCA / IForest / HBOS
            f = np.load(self.path.parent / fnorm)
            self.fmu, self.fsd = f["mu"].astype(np.float32), f["sd"].astype(np.float32)
        self.pending = None

    def step(self, x_raw):
        x = ((x_raw[self.keep] - self.mu[self.keep]) / self.sd[self.keep]).astype(np.float32)
        self.buf = np.roll(self.buf, -1, 0)
        self.buf[-1] = x
        k = self.kind
        if k == "movingvar":
            return float(self.buf.var(0).mean())
        if k == "sqdiff":
            return float(((self.buf[-1] - self.buf[-2]) ** 2).mean())
        if k in ("pca", "hbos", "iforest", "lightgbm", "extratrees"):
            f = win_features(self.buf)
            if self.fmu is not None:
                f = (f - self.fmu) / self.fsd
            f = f[None].astype(np.float32)
            if k == "pca":
                c, mu = self.p["components"], self.p["mean"]
                z = (f - mu) @ c.T
                return float(((f - (z @ c + mu)) ** 2).mean())
            if k == "hbos":
                s = 0.0
                for j in range(f.shape[1]):
                    e, lp = self.p["edges"][j], self.p["logp"][j]
                    v = f[0, j]
                    s -= (lp.min() - 1.0) if (v < e[0] or v > e[-1]) else lp[min(max(np.searchsorted(e, v, "right") - 1, 0), len(lp) - 1)]
                return s
            if k == "iforest":
                return float(-self.m.score_samples(f)[0])
            if k == "lightgbm":
                return float(self.m.predict(f, raw_score=True)[0])
            return float(self.m.predict_proba(f)[0, 1])
        if k == "tspulse":
            # run the model every H steps; in between, compare new samples with the stored forecast
            if self.t % H == 0 or self.pending is None:
                # context = the 512 samples before the current one; the forecast covers the current step
                self.pending = np.asarray(self._run(np.concatenate([self.buf[:1], self.buf[:-1]], 0)[None]))[0]
            self.t += 1
            return float(((self.pending[(self.t - 1) % H] - x) ** 2).mean())
        return float(np.asarray(self._run(self.buf[None])).ravel()[0])

    def _run(self, a):
        if hasattr(self, "sess"):
            return self.sess.run(None, {self.inp: a.astype(np.float32)})[0]
        import torch
        with torch.no_grad():
            return self.m(torch.from_numpy(a.astype(np.float32))).numpy()


def run(a):
    pkg = Path(a.pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    data = np.load(pkg / "sample.npy")  # one real test flight, (T, C_all)
    out = dict(host=platform.node(), machine=platform.machine(), python=sys.version.split()[0], ds=meta["ds"],
               results=[])
    try:
        out["temp_start"] = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
    except OSError:
        pass
    for thr in a.threads:
        os.environ["OMP_NUM_THREADS"] = str(thr)
        for item in meta["models"]:
            if a.only and item["method"] not in a.only:
                continue
            # stream the sample flight from the start and compare the last 64 scores with the offline scores
            chk = Stream(item["kind"], pkg / item["file"], thr, item.get("fnorm"))
            got = np.array([chk.step(x) for x in data])[-64:]
            ref = np.asarray(item["ref_scores"])
            rel = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-12))
            del chk
            st = Stream(item["kind"], pkg / item["file"], thr, item.get("fnorm"))
            for i in range(min(200, len(data))):  # warm-up
                st.step(data[i % len(data)])
            lat = []
            for i in range(a.steps):
                x = data[i % len(data)]
                t0 = time.perf_counter()
                st.step(x)
                lat.append(time.perf_counter() - t0)
            lat = np.array(lat) * 1e3
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            r = dict(method=item["method"], threads=thr, median_ms=float(np.median(lat)),
                     p95_ms=float(np.percentile(lat, 95)), mean_ms=float(lat.mean()), max_rss_mb=rss,
                     file_kb=round((pkg / item["file"]).stat().st_size / 1024, 1), match_rel_err=rel)
            out["results"].append(r)
            print(f"{thr}t {item['method']:11s} median {r['median_ms']:.3f} ms  p95 {r['p95_ms']:.3f} ms  "
                  f"rss {rss:.0f} MB  match {rel:.1e}", flush=True)
            del st
    try:
        out["temp_end"] = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
    except OSError:
        pass
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--pkg", required=True)
    r.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    r.add_argument("--steps", type=int, default=2000)
    r.add_argument("--only", nargs="*")
    r.add_argument("--out", default="edge_results.json")
    e = sp.add_parser("export")
    e.add_argument("--ds", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from edge_export import export  # noqa: E402
        export(a.ds, a.out)
