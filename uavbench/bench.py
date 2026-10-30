"""Benchmark core: loading, grouped splits, label budgets, normalisation, causal windows, features, metrics.

The score at time t uses samples up to t only (window [t-W+1, t]; the first sample is repeated at the start),
and channels are standardised with training-set statistics.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(os.environ.get("UAVBENCH_DATA", Path(__file__).resolve().parents[1] / "data/processed"))
CH_K = ["p", "q", "r", "ax", "ay", "az", "roll", "pitch", "yaw", "vx", "vy", "vz"]
# Set F drops quantities that depend on the flight site (heading, horizontal position);
# barometric altitude is made relative to take-off.
F_EXCLUDE = {"yaw", "pos_x", "pos_y"}
# ALFA injects failures through the command itself (throttle = 0, fixed servo command), so these channels
# carry the label. Set F removes them; set FA keeps them for the ablation.
ALFA_INJECT = {"throttle", "rc0", "rc1", "rc2", "rc3"}
HZ = 10
GROUP_UAVSEAD = "session"  # "log" reproduces the log-level split of the supplementary comparison


@dataclass
class Flight:
    fid: str
    group: str
    ftype: str
    X: np.ndarray    # (T, C) float32
    y: np.ndarray    # (T,) int8
    air: np.ndarray  # (T,) bool, samples that are scored

    @property
    def anomalous(self):
        return bool(self.y[self.air].any())


# rfly_ctrl_lxl fault code -> fault type (taken from the log, not from the folder name)
RFLY_ID = {123450: "Motor", 123452: "Accel", 123453: "Gyro", 123454: "Baro", 123455: "Mag", 123456: "GPS"}


def _rfly_ftype(row):
    ids = json.loads(row["fault_ids"])
    return "+".join(RFLY_ID.get(i, str(i)) for i in ids) if ids else "No_Fault"


def load(ds, chset="F", hz=10):
    """Return (flights, channels). RflyMAD is stored at 50 Hz and decimated by taking every 50/hz-th sample."""
    meta = pd.read_csv(PROC / ds / "meta.csv")
    if "error" in meta:
        meta = meta[meta["error"].isna()]
    if ds == "rflymad":  # runs stopped before the fault was injected are not treated as normal flights
        meta = meta[(meta["folder"] == "Real-No_Fault") | (meta["fault_ids"] != "[]")]
    if ds == "uavsead":  # anomaly class without any annotated range: flight-level label only, dropped
        meta = meta[(meta["fault_type"] == "Normal") | meta["onset_s"].notna()]
    flights, chans = [], None
    for _, r in meta.iterrows():
        df = pd.read_parquet(PROC / ds / "flights" / f"{r.flight_id}.parquet")
        if ds == "rflymad":
            df = df.iloc[::50 // hz].reset_index(drop=True)
            group, ftype = r.flight_id, _rfly_ftype(r)
        elif ds == "alfa":
            group, ftype = r.group, r.fault_type
        else:  # UAV-SEAD: group by recording session (date / session folder)
            group = r.flight_id.rsplit("__", 1)[0] if "__" in r.flight_id else "root"
            if GROUP_UAVSEAD == "log":
                group = r.flight_id
            ftype = r.fault_type
        if "baro_alt" in df:
            df["baro_alt"] = df["baro_alt"] - df["baro_alt"].iloc[0]
        cols = [c for c in df.columns if c not in ("t", "label", "in_air")]
        if chset == "K":
            cols = CH_K
        else:
            cols = [c for c in cols if c not in F_EXCLUDE]
            if ds == "alfa" and chset == "F":
                cols = [c for c in cols if c not in ALFA_INJECT]
        if chans is None:
            chans = cols
        assert cols == chans, (r.flight_id, set(cols) ^ set(chans))
        air = df["in_air"].to_numpy(bool) if "in_air" in df else np.ones(len(df), bool)
        flights.append(Flight(r.flight_id, str(group), str(ftype), df[cols].to_numpy(np.float32),
                              df["label"].to_numpy(np.int8), air))
    return flights, chans


# ---------------------------------------------------------------- splits and label budget

def split(flights, k):
    """Grouped 5-fold CV stratified by fault type (fixed shuffle): fold k = test, fold k+1 = val, rest = train."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = [f.ftype for f in flights]
    g = [f.group for f in flights]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g))
    te, va = set(folds[k][1]), set(folds[(k + 1) % 5][1])
    tr = [i for i in range(len(flights)) if i not in te and i not in va]
    out = dict(train=tr, val=sorted(va), test=sorted(te))
    gs = {k: {flights[i].group for i in v} for k, v in out.items()}
    assert not (gs["train"] & gs["val"]) and not (gs["train"] & gs["test"]) and not (gs["val"] & gs["test"])
    return out


def budget(flights, idx, frac, seed, min_each=2):
    """Labelled subset: whole groups in a fixed random order (so budgets are nested) until at least
    ceil(frac * n) flights, min_each faulty and min_each normal flights are included."""
    if frac >= 1:
        return list(idx)
    groups = sorted({flights[i].group for i in idx})
    perm = [groups[i] for i in np.random.default_rng(1000 + seed).permutation(len(groups))]
    by_g = {}
    for i in idx:
        by_g.setdefault(flights[i].group, []).append(i)
    need = int(np.ceil(frac * len(idx)))
    keep, n_a, n_n = [], 0, 0
    for g in perm:
        keep += by_g[g]
        n_a += sum(flights[i].anomalous for i in by_g[g])
        n_n += sum(not flights[i].anomalous for i in by_g[g])
        if len(keep) >= need and n_a >= min_each and n_n >= min_each:
            break
    return sorted(keep)


# ---------------------------------------------------------------- normalisation, windows, features

class Norm:
    def fit(self, Xs):
        A = np.concatenate(Xs).astype(np.float64)
        self.mu, self.sd = A.mean(0), A.std(0)
        self.keep = np.flatnonzero(self.sd > 1e-8)  # drop channels that are constant in training
        return self

    def __call__(self, X):
        return ((X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]).astype(np.float32)


def windows(X, W):
    """(T, C) -> (T, W, C) with the window ending at t; the first sample is repeated at the start."""
    Xp = np.concatenate([np.repeat(X[:1], W - 1, 0), X], 0)
    return np.lib.stride_tricks.sliding_window_view(Xp, W, axis=0).transpose(0, 2, 1)


FEAT_NAMES = ["mean", "std", "min", "max", "last", "delta", "mad", "maxad"]


def win_features(X, W):
    """Eight statistics per channel over the causal window: (T, 8C)."""
    w = windows(X, W)
    d = np.abs(np.diff(w, axis=1))
    f = [w.mean(1), w.std(1), w.min(1), w.max(1), w[:, -1], w[:, -1] - w[:, 0], d.mean(1), d.max(1)]
    return np.concatenate(f, 1).astype(np.float32)


# ---------------------------------------------------------------- metrics

def evaluate(scores, ys, thr=None, sliding=None):
    """Concatenate the test flights (normal flights included, so their alarms count) and compute VUS-PR, VUS-ROC,
    AUC-PR and AUC-ROC with the TSB-AD implementation; no point adjustment. With thresholds {tag: value},
    also the median detection delay, miss rate and share of normal flights with any alarm."""
    from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve
    sliding = sliding or HZ  # VUS buffer of up to 1 s
    s = np.concatenate(scores).astype(np.float64)
    y = np.concatenate(ys).astype(int)
    s = np.nan_to_num(s, nan=np.nanmin(s), posinf=np.nanmax(s[np.isfinite(s)]))
    g = basic_metricor()
    *_, vus_roc, vus_pr = generate_curve(y, s, sliding, "opt", 250)
    out = {"VUS-PR": float(vus_pr), "VUS-ROC": float(vus_roc), "AUC-PR": float(g.metric_PR(y, s)),
           "AUC-ROC": float(g.metric_ROC(y, s))}
    for tag, t in (thr or {}).items():
        out.update({f"{k}@{tag}": v for k, v in _operating(scores, ys, t).items()})
    return out


def _operating(scores, ys, thr):
    delays, miss, fa = [], 0, []
    for sc, yy in zip(scores, ys):
        if yy.any():
            on = np.flatnonzero(yy)
            hit = np.flatnonzero(sc[on[0]:on[-1] + 1] > thr)
            if len(hit):
                delays.append(hit[0] / HZ)
            else:
                miss += 1
        else:
            fa.append(bool((sc > thr).any()))
    n_anom = sum(bool(yy.any()) for yy in ys)
    return dict(delay_med_s=float(np.median(delays)) if delays else float("nan"),
                miss_rate=miss / max(1, n_anom), false_alarm_flights=float(np.mean(fa)) if fa else float("nan"))


def threshold_at_fpr(scores, ys, fpr=0.001, guard=None):
    """(1 - fpr) quantile of validation scores at normal samples, excluding +-guard steps around fault intervals
    where the vehicle is still recovering after the fault is switched off."""
    guard = guard or 2 * HZ
    keep = []
    for sc, y in zip(scores, ys):
        bad = y.astype(bool).copy()
        if y.any():
            k = np.ones(2 * guard + 1)
            bad = np.convolve(y.astype(float), k, mode="same") > 0
        keep.append(sc[~bad])
    return float(np.quantile(np.concatenate(keep), 1 - fpr))


def normal_segments(X, y, min_len, guard=None):
    """Contiguous normal segments (away from faults by +-guard) of at least min_len samples."""
    guard = guard or 2 * HZ
    bad = np.convolve(y.astype(float), np.ones(2 * guard + 1), mode="same") > 0 if y.any() else np.zeros(len(y), bool)
    out, start = [], None
    for t, b in enumerate(np.r_[bad, True]):
        if not b and start is None:
            start = t
        elif b and start is not None:
            if t - start >= min_len:
                out.append(X[start:t])
            start = None
    return out


def save_record(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=float))
