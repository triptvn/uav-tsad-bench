"""Run the benchmark: one record JSON per (dataset, channel set, detector, fold, label budget).

Existing records are skipped, so the script can be re-run after an interruption. Training-free and unsupervised
detectors are fitted once per fold (no labels); the label budget only changes their validation threshold.

  python tools/run_bench.py --ds rflymad --budgets 0.05 0.2 1.0
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavbench import bench, methods  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True, choices=["alfa", "rflymad", "uavsead"])
ap.add_argument("--chset", default="F", choices=["F", "K", "FA"])
ap.add_argument("--methods", nargs="+", default=["all"])
ap.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.2, 1.0])
ap.add_argument("--out", default="results/records")
ap.add_argument("--tag", default="")
ap.add_argument("--hz", type=int, default=10, help="50 is available for RflyMAD only")
ap.add_argument("--window", type=int, default=32)
ap.add_argument("--uavsead-group", default="session", choices=["session", "log"])
ap.add_argument("--unsup-normal", action="store_true",
                help="train unsupervised detectors on normal segments of the labelled flights")
a = ap.parse_args()

names = list(methods.REGISTRY) if a.methods == ["all"] else a.methods
bench.HZ = a.hz
bench.GROUP_UAVSEAD = a.uavsead_group
methods.W = a.window
flights, chans = bench.load(a.ds, a.chset, hz=a.hz)
print(f"{a.ds}/{a.chset}: {len(flights)} flights, {len(chans)} channels, "
      f"{sum(len(f.y) for f in flights)} samples, faulty flights {sum(f.anomalous for f in flights)}", flush=True)
out_root = Path(a.out) / f"{a.ds}_{a.chset}{a.tag}"


def sub(idx, norm, labeled=True):
    return dict(X=[norm(flights[i].X) for i in idx], y=[flights[i].y for i in idx] if labeled else None,
                air=[flights[i].air for i in idx])


def masked(scores, d):
    """Score in-air samples only; detectors still see the whole sequence."""
    return [s[m] for s, m in zip(scores, d["air"])], [y[m] for y, m in zip(d["y"], d["air"])]


for k in a.splits:
    sp = bench.split(flights, k)
    norm = bench.Norm().fit([flights[i].X for i in sp["train"]])
    te = sub(sp["test"], norm)
    lab = {b: (bench.budget(flights, sp["train"], b, k), bench.budget(flights, sp["val"], b, k)) for b in a.budgets}
    for name in names:
        cls = methods.REGISTRY[name]
        todo = [b for b in a.budgets if not (out_root / name / f"s{k}_b{b:g}.json").exists()]
        fitted = None
        for b in todo:
            tr_idx, va_idx = lab[b]
            if cls.kind == "sup" or fitted is None or (a.unsup_normal and cls.kind == "unsup"):
                m = cls(seed=k) if cls.kind != "none" else cls()
                if cls.kind == "sup":
                    _, t_fit = methods.timed(m.fit, sub(tr_idx, norm), sub(va_idx, norm))
                elif a.unsup_normal and cls.kind == "unsup":
                    def seg(idx):
                        return dict(X=[s for i in idx for s in bench.normal_segments(
                            norm(flights[i].X), flights[i].y, methods.W + 8)], y=None)
                    _, t_fit = methods.timed(m.fit, seg(tr_idx), seg(va_idx))
                else:
                    _, t_fit = methods.timed(m.fit, sub(sp["train"], norm, False), sub(sp["val"], norm, False))
                fitted = (m, t_fit)
            m, t_fit = fitted
            va = sub(va_idx, norm)
            vs = masked([m.score(X) for X in va["X"]], va)
            thr = {"fpr0.1": bench.threshold_at_fpr(*vs, 0.001), "fpr1": bench.threshold_at_fpr(*vs, 0.01)}
            t = time.time()
            sc = [m.score(X) for X in te["X"]]
            t_score = time.time() - t
            res = bench.evaluate(*masked(sc, te), thr=thr)
            rec = dict(ds=a.ds, chset=a.chset, method=name, kind=cls.kind, split=k, budget=b, window=methods.W,
                       hz=a.hz, unsup_normal=bool(a.unsup_normal and cls.kind == "unsup"), tag=a.tag,
                       channels=[chans[j] for j in norm.keep], n_channels=int(len(norm.keep)),
                       n_train_flights=len(sp["train"]), n_train_labeled=len(tr_idx), n_val_labeled=len(va_idx),
                       n_test_flights=len(sp["test"]), n_test_points=int(sum(m_.sum() for m_ in te["air"])),
                       test=res, thr=thr, fit_s=round(t_fit, 2),
                       score_us_per_point=round(1e6 * t_score / sum(len(y) for y in te["y"]), 2),
                       n_params=m.n_params(), host=platform.node(), hist=getattr(m, "hist", None))
            bench.save_record(out_root / name / f"s{k}_b{b:g}.json", rec)
            print(f"fold {k} budget {b:g} {name:11s} VUS-PR {res['VUS-PR']:.3f} AUC-PR {res['AUC-PR']:.3f} "
                  f"labelled {len(tr_idx)}/{len(sp['train'])}", flush=True)
