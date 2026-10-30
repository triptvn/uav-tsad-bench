"""Build the paper tables (LaTeX + CSV) and the quality-latency figure from the record JSONs.

Outputs: results/tables/{tab_quality,tab_edge,tab_ablation}.tex, results/figures/fig_frontier.{pdf,png},
         results/summary/{quality,edge,ablation}.csv
  python tools/make_tables.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results"
TAB = EXP / "tables"
FIG = EXP / "figures"
SUM = EXP / "summary"
for d in (TAB, FIG, SUM):
    d.mkdir(parents=True, exist_ok=True)

FAMILY = {  # detector family
    "MovingVar": "Training-free", "SqDiff": "Training-free",
    "PCA": "Unsupervised", "IForest": "Unsupervised", "HBOS": "Unsupervised", "USAD": "Unsupervised",
    "FITS": "Unsupervised", "TSPulse": "TSFM zero-shot",
    "LightGBM": "Supervised", "ExtraTrees": "Supervised", "CNN": "Supervised", "TCN": "Supervised", "LSTM": "Supervised"}
ORDER = list(FAMILY)
FAM_ORDER = ["Training-free", "Unsupervised", "TSFM zero-shot", "Supervised"]
RANDOM = {"alfa": 0.164, "rflymad": 0.303, "uavsead": 0.051}  # share of anomalous in-air samples = random AUC-PR


def records(pattern):
    rows = []
    for f in glob.glob(str(EXP / "records" / pattern / "*" / "*.json")):
        x = json.load(open(f))
        t = x["test"]
        rows.append(dict(cfg=Path(f).parts[-3], ds=x["ds"], m=x["method"], b=x["budget"], s=x["split"],
                         vus=t["VUS-PR"], aupr=t["AUC-PR"], roc=t["AUC-ROC"],
                         delay=t.get("delay_med_s@fpr1"), miss=t.get("miss_rate@fpr1"),
                         fa=t.get("false_alarm_flights@fpr1"), params=x["n_params"]))
    return pd.DataFrame(rows)


def fmt(mu, sd, bold=False):
    s = f"{mu:.3f}" if np.isnan(sd) else f"{mu:.3f}\\,\\scriptsize{{$\\pm${sd:.3f}}}"
    return f"\\textbf{{{mu:.3f}}}" + s[5:] if bold else s


# ------------------------------------------------------------------ quality table
d = pd.concat([records("alfa_F"), records("rflymad_F"), records("uavsead_F")])
cols = [("alfa", 1.0), ("rflymad", 0.05), ("rflymad", 0.2), ("rflymad", 1.0),
        ("uavsead", 0.05), ("uavsead", 0.2), ("uavsead", 1.0)]
g = d.groupby(["m", "ds", "b"]).vus.agg(["mean", "std", "count"])
g.to_csv(SUM / "quality.csv")
best = {c: g.xs(c[0], level="ds").xs(c[1], level="b")["mean"].max() for c in cols}
lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         r" & ALFA & \multicolumn{3}{c}{RflyMAD-Real} & \multicolumn{3}{c}{UAV-SEAD} \\",
         r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Labels & 100\% & 5\% & 20\% & 100\% & 5\% & 20\% & 100\% \\", r"\midrule",
         r"\textit{Random} & " + " & ".join(f"{RANDOM[c[0]]:.3f}" for c in cols) + r" \\"]
for fam in FAM_ORDER:
    lines.append(r"\midrule")
    for m in [x for x in ORDER if FAMILY[x] == fam]:
        cells = []
        for c in cols:
            r = g.loc[(m, c[0], c[1])]
            cells.append(fmt(r["mean"], r["std"], bold=np.isclose(r["mean"], best[c])))
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_quality.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ Raspberry Pi 5 cost table
e = []
for f in glob.glob(str(EXP / "edge_pi5/*.json")):
    x = json.load(open(f))
    r = x["results"][0]
    e.append(dict(ds=x["ds"], m=r["method"], t=r["threads"], mean=r["mean_ms"], med=r["median_ms"], p95=r["p95_ms"],
                  rss=r["max_rss_mb"], kb=r["file_kb"], err=r["match_rel_err"]))
e = pd.DataFrame(e)
e.to_csv(SUM / "edge.csv", index=False)
params = d[(d.ds == "rflymad") & (d.b == 1.0) & (d.s == 0)].set_index("m").params
er = e[(e.ds == "rflymad") & (e.t == 1)].set_index("m")
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Method & Params & Mean (ms) & p95 (ms) & RSS (MB) & Budget \\", r"\midrule"]
for fam in FAM_ORDER:
    for m in [x for x in ORDER if FAMILY[x] == fam]:
        r = er.loc[m]
        p = params.get(m, 0)
        ps = "--" if p == 0 else (f"{p / 1e6:.2f}\\,M" if p >= 1e6 else f"{p / 1e3:.1f}\\,k")
        lines.append(f"{m} & {ps} & {r['mean']:.2f} & {r['p95']:.2f} & {r['rss']:.0f} & {r['mean']:.1f}\\% \\\\")
    lines.append(r"\midrule" if fam != FAM_ORDER[-1] else r"\bottomrule")
lines += [r"\end{tabular}"]
(TAB / "tab_edge.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ ablation table
a = pd.concat([records(p) for p in ("rflymad_F", "rflymad_K", "rflymad_F_normal", "rflymad_F_w16", "rflymad_F_w64",
                                    "rflymad_F_hz50", "alfa_F", "alfa_FA")])
ga = a.groupby(["cfg", "m", "b"]).vus.mean().round(3)
ga.to_csv(SUM / "ablation.csv")


def v(cfg, m, b=1.0, folds=None):
    t = a[(a.cfg == cfg) & (a.m == m) & (a.b == b)]
    if folds is not None:
        t = t[t.s.isin(folds)]
    return t.vus.mean()


rows = [
    ("Channel set K (12 ch) vs F", "LightGBM", v("rflymad_K", "LightGBM"), v("rflymad_F", "LightGBM")),
    ("", "TCN", v("rflymad_K", "TCN"), v("rflymad_F", "TCN")),
    ("Unsup. trained on normal segments", "PCA", v("rflymad_F_normal", "PCA"), v("rflymad_F", "PCA")),
    ("", "USAD", v("rflymad_F_normal", "USAD"), v("rflymad_F", "USAD")),
    ("Window 16 / 64 vs 32", "LightGBM", f"{v('rflymad_F_w16', 'LightGBM'):.3f} / {v('rflymad_F_w64', 'LightGBM'):.3f}",
     v("rflymad_F", "LightGBM")),
    ("50 Hz vs 10 Hz (folds 0--1)", "LightGBM", v("rflymad_F_hz50", "LightGBM", folds=[0, 1]),
     v("rflymad_F", "LightGBM", folds=[0, 1])),
    ("ALFA with injection channels", "TCN", v("alfa_FA", "TCN"), v("alfa_F", "TCN")),
]
lines = [r"\begin{tabular}{llcc}", r"\toprule", r"Variant & Method & Variant & Main \\", r"\midrule"]
for name, m, x, y in rows:
    xs = x if isinstance(x, str) else f"{x:.3f}"
    lines.append(f"{name} & {m} & {xs} & {y:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_ablation.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ quality-latency figure (RflyMAD-Real)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"Training-free": "#2a78d6", "Unsupervised": "#eb6834", "TSFM zero-shot": "#1baf7a", "Supervised": "#eda100"}
MK = {"Training-free": "s", "Unsupervised": "o", "TSFM zero-shot": "D", "Supervised": "^"}
q = d[d.ds == "rflymad"].groupby(["m", "b"]).vus.mean()
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.axhline(RANDOM["rflymad"], color="#8d8d8d", lw=0.8, ls="--", zorder=0)
ax.text(0.017, RANDOM["rflymad"] - 0.035, "random", color="#5f5f5f", fontsize=6)
ax.axvline(100, color="#8d8d8d", lw=0.8, ls=":", zorder=0)
ax.text(85, 0.215, "100 ms budget", color="#5f5f5f", fontsize=6, ha="right", va="bottom")
for fam in FAM_ORDER:
    ms = [m for m in ORDER if FAMILY[m] == fam]
    x = [er.loc[m, "mean"] for m in ms]
    ax.scatter(x, [q[(m, 1.0)] for m in ms], s=26, marker=MK[fam], color=COL[fam], edgecolor="white", lw=0.8,
               label=fam, zorder=3)
    if fam == "Supervised":
        ax.scatter(x, [q[(m, 0.05)] for m in ms], s=26, marker=MK[fam], facecolor="white", edgecolor=COL[fam], lw=1.0,
                   zorder=3, label="Supervised, 5% labels")
        for m, xx in zip(ms, x):
            ax.plot([xx, xx], [q[(m, 0.05)], q[(m, 1.0)]], color=COL[fam], lw=0.8, alpha=0.6, zorder=2)
off = {"LightGBM": (4, 2), "ExtraTrees": (4, -2), "CNN": (-4, -9), "TCN": (4, -1), "LSTM": (-4, 4),
       "MovingVar": (4, -8), "SqDiff": (4, 2), "PCA": (0, 6), "IForest": (4, 2), "HBOS": (4, 2), "USAD": (4, -8),
       "FITS": (-4, 4), "TSPulse": (4, 2)}
for m in ORDER:
    dx, dy = off[m]
    ax.annotate(m, (er.loc[m, "mean"], q[(m, 1.0)]), xytext=(dx, dy), textcoords="offset points", fontsize=5.5,
                ha="left" if dx > 0 else ("center" if dx == 0 else "right"), color="#2b2b2b")
ax.set_xscale("log")
ax.set_xlim(0.015, 150)
ax.set_ylim(0.2, 1.0)
ax.set_xlabel("Mean latency per step on Raspberry Pi 5, 1 thread (ms)")
ax.set_ylabel("VUS-PR (RflyMAD-Real)")
ax.grid(True, which="major", color="#e6e6e6", lw=0.5)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.47), fontsize=5.5, frameon=False, handletextpad=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_frontier.pdf")
fig.savefig(FIG / "fig_frontier.png", dpi=220)
print("tables:", sorted(p.name for p in TAB.glob("*.tex")), "| figure: fig_frontier")
