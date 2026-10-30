"""Build the model package for edge_bench.py: train every detector on fold 0 with all labels, then export
networks to ONNX (output = score), trees to joblib, PCA/HBOS to npz, and FFT-based models to TorchScript.
The last 64 offline scores of one test flight are stored so the device can check its streamed scores."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavbench import bench, methods  # noqa: E402

KIND = {"MovingVar": "movingvar", "SqDiff": "sqdiff", "PCA": "pca", "IForest": "iforest", "HBOS": "hbos",
        "LightGBM": "lightgbm", "ExtraTrees": "extratrees", "USAD": "net", "FITS": "net", "CNN": "net",
        "TCN": "net", "LSTM": "net", "TSPulse": "tspulse"}


def _score_module(m):
    """Wrap a network as a module (1, W, C) -> score (1,)."""
    import torch
    nn = torch.nn

    for p in m.net.parameters():
        p.requires_grad_(False)

    class S(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = m.net  # registers the parameters so ONNX stores them as initializers

        def forward(s, x):
            return m._score_batch(x)
    return S().eval()


def export(ds, out, chset="F"):
    import joblib
    import torch
    out = Path(out) / ds
    out.mkdir(parents=True, exist_ok=True)
    flights, chans = bench.load(ds, chset)
    sp = bench.split(flights, 0)
    norm = bench.Norm().fit([flights[i].X for i in sp["train"]])
    # keep float64 as offline: float32 statistics move HBOS bin assignments by up to 17 %
    np.savez(out / "norm.npz", mu=norm.mu, sd=norm.sd, keep=norm.keep)

    def sub(idx, lab=True):
        return dict(X=[norm(flights[i].X) for i in idx], y=[flights[i].y for i in idx] if lab else None)
    tr_l, va_l, tr_u, va_u = sub(sp["train"]), sub(sp["val"]), sub(sp["train"], False), sub(sp["val"], False)
    te0 = max(sp["test"], key=lambda i: len(flights[i].y) if flights[i].anomalous else 0)
    np.save(out / "sample.npy", flights[te0].X.astype(np.float32))
    Xs = norm(flights[te0].X)
    items = []
    for name, cls in methods.REGISTRY.items():
        m = cls(seed=0) if cls.kind != "none" or name == "TSPulse" else cls()
        if name == "TSPulse":
            m.dev = "cpu"
        m.fit(tr_l, va_l) if cls.kind == "sup" else m.fit(tr_u, va_u)
        ref = m.score(Xs)
        it = dict(method=name, kind=KIND[name], n_params=m.n_params())
        if name in ("MovingVar", "SqDiff"):
            it["file"] = "norm.npz"
        elif name in ("PCA", "HBOS", "IForest"):
            np.savez(out / f"{name}_fnorm.npz", mu=m.mu, sd=m.sd)
            it["fnorm"] = f"{name}_fnorm.npz"
            if name == "PCA":
                np.savez(out / "PCA.npz", components=m.m.components_.astype(np.float32),
                         mean=m.m.mean_.astype(np.float32))
                it["file"] = "PCA.npz"
            elif name == "HBOS":
                e = np.empty(len(m.edges), object)
                lp = np.empty(len(m.logp), object)
                e[:], lp[:] = m.edges, m.logp
                np.savez(out / "HBOS.npz", edges=e, logp=lp)
                it["file"] = "HBOS.npz"
            else:
                joblib.dump(m.m, out / "IForest.joblib")
                it["file"] = "IForest.joblib"
        elif name in ("LightGBM", "ExtraTrees"):
            joblib.dump(m.m, out / f"{name}.joblib")
            it["file"] = f"{name}.joblib"
        elif name == "TSPulse":
            class F(torch.nn.Module):
                def __init__(s):
                    super().__init__()
                    s.n = m.net

                def forward(s, x):
                    return s.n(x)["forecast_output"][:, :methods.TSPulseZS.H]
            f = F().eval()
            dummy = torch.zeros(1, methods.TSPulseZS.CTX, Xs.shape[1])
            try:
                torch.onnx.export(f, dummy, out / "TSPulse.onnx", input_names=["x"], output_names=["y"],
                                  opset_version=17, dynamo=False)
                it["file"] = "TSPulse.onnx"
            except Exception as e:  # rfft has no ONNX opset-17 export -> TorchScript
                print("TSPulse: ONNX export failed, using TorchScript:", repr(e)[:200])
                torch.jit.trace(f, dummy).save(str(out / "TSPulse.pt"))
                it["file"] = "TSPulse.pt"
        else:
            m.net.cpu()
            m.dev = "cpu"
            s = _score_module(m)
            dummy = torch.zeros(1, methods.W, Xs.shape[1])
            try:
                torch.onnx.export(s, dummy, out / f"{name}.onnx", input_names=["x"], output_names=["y"],
                                  opset_version=17, dynamo=False)
                it["file"] = f"{name}.onnx"
            except Exception as e:  # e.g. FITS uses rfft -> TorchScript
                print(f"{name}: ONNX export failed ({type(e).__name__}), using TorchScript", flush=True)
                (out / f"{name}.onnx").unlink(missing_ok=True)
                torch.jit.trace(s, dummy).save(str(out / f"{name}.pt"))
                it["file"] = f"{name}.pt"
        it["ref_scores"] = ref[-64:].tolist()
        items.append(it)
        print(f"{name:11s} -> {it['file']}  params {it['n_params']}", flush=True)
    (out / "meta.json").write_text(json.dumps(dict(ds=ds, chset=chset, channels=chans, window=methods.W,
                                                   sample_flight=flights[te0].fid, models=items), indent=1))
