"""The 13 detectors. Common interface: fit(tr, va), then score(X) -> (T,), where the score at t uses X[:t+1] only.

tr, va: dict(X=[(T, C) normalised arrays], y=[(T,) labels] or None when unlabelled).
kind: none (no training), unsup (all training flights, no labels), sup (labelled flights only).
"""
import time

import numpy as np

from .bench import win_features, windows

W = 32  # 3.2 s at 10 Hz


class Method:
    kind = "none"
    name = "base"

    def fit(self, tr, va):
        return self

    def n_params(self):
        return 0


# ---------------------------------------------------------------- one-liners (Zhu et al., ICLR 2026)

class MovingVar(Method):
    name = "MovingVar"

    def score(self, X):
        return windows(X, W).var(1).mean(1)


class SqDiff(Method):
    name = "SqDiff"

    def score(self, X):
        d = np.diff(X, axis=0, prepend=X[:1])
        return (d ** 2).mean(1)


# ---------------------------------------------------------------- classical detectors on window statistics

def _stack_feats(Xs, ys=None, stride=4, cap=200_000, seed=0):
    F = [win_features(X, W)[::stride] for X in Xs]
    Y = [y[::stride] for y in ys] if ys is not None else None
    F = np.concatenate(F)
    Y = np.concatenate(Y) if Y is not None else None
    if len(F) > cap:
        idx = np.random.default_rng(seed).choice(len(F), cap, replace=False)
        F, Y = F[idx], (Y[idx] if Y is not None else None)
    return F, Y


class _FeatUnsup(Method):
    kind = "unsup"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, tr, va):
        F, _ = _stack_feats(tr["X"], seed=self.seed)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-6
        self._fit((F - self.mu) / self.sd)
        return self

    def score(self, X):
        return self._score((win_features(X, W) - self.mu) / self.sd)


class PCAAD(_FeatUnsup):
    name = "PCA"

    def _fit(self, F):
        from sklearn.decomposition import PCA
        self.m = PCA(n_components=0.95, random_state=self.seed).fit(F)

    def _score(self, F):
        R = self.m.inverse_transform(self.m.transform(F))
        return ((F - R) ** 2).mean(1)

    def n_params(self):
        return int(self.m.components_.size + self.m.mean_.size)


class IForestAD(_FeatUnsup):
    name = "IForest"

    def _fit(self, F):
        from sklearn.ensemble import IsolationForest
        self.m = IsolationForest(n_estimators=100, max_samples=256, random_state=self.seed, n_jobs=8).fit(F)

    def _score(self, F):
        return -self.m.score_samples(F)

    def n_params(self):
        return int(sum(e.tree_.node_count for e in self.m.estimators_) * 3)


class HBOSAD(_FeatUnsup):
    name = "HBOS"

    def _fit(self, F, bins=10):
        self.edges = [np.histogram_bin_edges(F[:, j], bins) for j in range(F.shape[1])]
        self.logp = []
        for j, e in enumerate(self.edges):
            h, _ = np.histogram(F[:, j], e)
            self.logp.append(np.log((h + 1) / (h.sum() + bins)))

    def _score(self, F):
        s = np.zeros(len(F))
        for j, (e, lp) in enumerate(zip(self.edges, self.logp)):
            k = np.clip(np.searchsorted(e, F[:, j], side="right") - 1, 0, len(lp) - 1)
            out = (F[:, j] < e[0]) | (F[:, j] > e[-1])
            s -= np.where(out, lp.min() - 1.0, lp[k])  # outside the training range: rarer than the rarest bin
        return s

    def n_params(self):
        return int(sum(len(e) + len(l) for e, l in zip(self.edges, self.logp)))


# ---------------------------------------------------------------- supervised tree ensembles

class _FeatSup(Method):
    kind = "sup"

    def __init__(self, seed=0):
        self.seed = seed


class LightGBMAD(_FeatSup):
    name = "LightGBM"

    def fit(self, tr, va):
        import lightgbm as lgb
        F, Y = _stack_feats(tr["X"], tr["y"], seed=self.seed)
        Fv, Yv = _stack_feats(va["X"], va["y"], seed=self.seed)
        self.m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                                    random_state=self.seed, n_jobs=8, verbose=-1)
        self.m.fit(F, Y, eval_set=[(Fv, Yv)], eval_metric="average_precision",
                   callbacks=[lgb.early_stopping(30, verbose=False)])
        return self

    def score(self, X):
        return self.m.predict(win_features(X, W), raw_score=True)  # log-odds do not saturate

    def n_params(self):
        d = self.m.booster_.dump_model()
        return int(sum(t["num_leaves"] * 2 - 1 for t in d["tree_info"]))


class ExtraTreesAD(_FeatSup):
    name = "ExtraTrees"

    def fit(self, tr, va):
        from sklearn.ensemble import ExtraTreesClassifier
        F, Y = _stack_feats(tr["X"], tr["y"], seed=self.seed)
        self.m = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                      random_state=self.seed, n_jobs=16).fit(F, Y)
        return self

    def score(self, X):
        return self.m.predict_proba(win_features(X, W))[:, 1]

    def n_params(self):
        return int(sum(e.tree_.node_count for e in self.m.estimators_) * 3)


# ---------------------------------------------------------------- neural detectors (PyTorch), one training loop

def _torch():
    import torch
    return torch


def _win_set(Xs, ys, stride, cap, seed):
    Wx = [windows(X, W)[::stride] for X in Xs]
    Wx = np.concatenate(Wx)
    Wy = np.concatenate([y[::stride] for y in ys]).astype(np.float32) if ys is not None else None
    if len(Wx) > cap:
        idx = np.random.default_rng(seed).choice(len(Wx), cap, replace=False)
        Wx, Wy = Wx[idx], (Wy[idx] if Wy is not None else None)
    return np.ascontiguousarray(Wx, dtype=np.float32), Wy


class _Net(Method):
    """Shared loop: AdamW, lr 1e-3, batch 256, at most 30 epochs, early stopping with patience 5 on validation."""
    epochs, patience, bs, lr = 30, 5, 256, 1e-3

    def __init__(self, seed=0, device=None):
        self.seed = seed
        torch = _torch()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))

    def _train(self, Xtr, Ytr, Xva, Yva):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = self._build(Xtr.shape[2]).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        best, bad, best_state = np.inf, 0, None
        Xt = torch.from_numpy(Xtr)
        Yt = torch.from_numpy(Ytr) if Ytr is not None else None
        g = torch.Generator().manual_seed(self.seed)
        self.hist = []
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.bs):
                b = perm[i:i + self.bs]
                xb = Xt[b].to(self.dev)
                yb = Yt[b].to(self.dev) if Yt is not None else None
                loss = self._loss(xb, yb, ep)
                opt.zero_grad()
                loss.backward()
                opt.step()
            v = self._val_loss(Xva, Yva)
            self.hist.append(float(v))
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()

    def _batched(self, Xw, fn):
        torch = _torch()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xw), 4096):
                out.append(fn(torch.from_numpy(np.ascontiguousarray(Xw[i:i + 4096])).to(self.dev)).float().cpu().numpy())
        return np.concatenate(out)

    def score(self, X):
        return self._batched(windows(X, W).astype(np.float32), self._score_batch)


class _SupNet(_Net):
    kind = "sup"

    def fit(self, tr, va):
        torch = _torch()
        Xtr, Ytr = _win_set(tr["X"], tr["y"], 4, 300_000, self.seed)
        Xva, Yva = _win_set(va["X"], va["y"], 4, 100_000, self.seed)
        pw = float(np.clip((1 - Ytr.mean()) / max(Ytr.mean(), 1e-6), 1, 50))
        self.crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=self.dev))
        self._train(Xtr, Ytr, Xva, Yva)
        return self

    def _loss(self, xb, yb, ep):
        return self.crit(self.net(xb).squeeze(-1), yb)

    def _val_loss(self, Xva, Yva):
        from sklearn.metrics import average_precision_score
        p = self._batched(Xva, self._score_batch)
        return -average_precision_score(Yva, p)  # early stopping on validation AUC-PR

    def _score_batch(self, xb):
        return self.net(xb).squeeze(-1)  # logits: sigmoid outputs saturate at 1.0 and break low-FPR thresholds


class CNN(_SupNet):
    """Two-layer LeNet-style 1D-CNN, as in the RflyMAD reference experiment."""
    name = "CNN"

    def _build(self, C):
        nn = _torch().nn

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.f = nn.Sequential(nn.Conv1d(C, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                                    nn.Conv1d(64, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                                    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 32), nn.ReLU(),
                                    nn.Linear(32, 1))

            def forward(s, x):
                return s.f(x.transpose(1, 2))
        return M()


class TCN(_SupNet):
    """Causal TCN with three blocks (dilation 1, 2, 4) and 32 channels; output taken at the last step."""
    name = "TCN"

    def _build(self, C):
        nn = _torch().nn
        F = nn.functional

        class Block(nn.Module):
            def __init__(s, cin, cout, d):
                super().__init__()
                s.pad = 2 * d
                s.c1 = nn.Conv1d(cin, cout, 3, dilation=d)
                s.c2 = nn.Conv1d(cout, cout, 3, dilation=d)
                s.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

            def forward(s, x):
                h = F.relu(s.c1(F.pad(x, (s.pad, 0))))
                h = F.relu(s.c2(F.pad(h, (s.pad, 0))))
                return F.relu(h + s.res(x))

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.b = nn.Sequential(Block(C, 32, 1), Block(32, 32, 2), Block(32, 32, 4))
                s.h = nn.Linear(32, 1)

            def forward(s, x):
                return s.h(s.b(x.transpose(1, 2))[:, :, -1])
        return M()


class LSTM(_SupNet):
    """Single-layer unidirectional LSTM with 64 units."""
    name = "LSTM"

    def _build(self, C):
        nn = _torch().nn

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.r = nn.LSTM(C, 64, batch_first=True)
                s.h = nn.Linear(64, 1)

            def forward(s, x):
                o, _ = s.r(x)
                return s.h(o[:, -1])
        return M()


class _UnsupNet(_Net):
    kind = "unsup"

    def fit(self, tr, va):
        Xtr, _ = _win_set(tr["X"], None, 4, 300_000, self.seed)
        Xva, _ = _win_set(va["X"], None, 4, 100_000, self.seed)
        self._train(Xtr, None, Xva, None)
        return self

    def _val_loss(self, Xva, Yva):
        return float(self._batched(Xva, self._score_batch).mean())


class USAD(_UnsupNet):
    """USAD (Audibert et al., KDD 2020): one encoder, two adversarially trained decoders; score = 0.5 e1 + 0.5 e2."""
    name = "USAD"

    def _build(self, C):
        nn = _torch().nn
        n, z = W * C, 64

        def enc():
            return nn.Sequential(nn.Linear(n, n // 2), nn.ReLU(), nn.Linear(n // 2, n // 4), nn.ReLU(),
                                 nn.Linear(n // 4, z), nn.ReLU())

        def dec():
            return nn.Sequential(nn.Linear(z, n // 4), nn.ReLU(), nn.Linear(n // 4, n // 2), nn.ReLU(),
                                 nn.Linear(n // 2, n), nn.Sigmoid())

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.e, s.d1, s.d2 = enc(), dec(), dec()

            def forward(s, x):
                z = s.e(x)
                w1 = s.d1(z)
                w2 = s.d2(z)
                w3 = s.d2(s.e(w1))
                return w1, w2, w3
        return M()

    def _prep(self, xb):
        # the decoders end in a sigmoid, so z-scores are mapped to [0, 1] with a fixed logistic function
        return xb.reshape(len(xb), -1).sigmoid()

    def _loss(self, xb, yb, ep):
        x = self._prep(xb)
        w1, w2, w3 = self.net(x)
        n = ep + 1
        l1 = (1 / n) * ((x - w1) ** 2).mean() + (1 - 1 / n) * ((x - w3) ** 2).mean()
        l2 = (1 / n) * ((x - w2) ** 2).mean() - (1 - 1 / n) * ((x - w3) ** 2).mean()
        return l1 + l2

    def _score_batch(self, xb):
        x = self._prep(xb)
        w1, _, w3 = self.net(x)
        return 0.5 * ((x - w1) ** 2).mean(1) + 0.5 * ((x - w3) ** 2).mean(1)

    def _val_loss(self, Xva, Yva):
        return float(self._batched(Xva, self._score_batch).mean())


class FITS(_UnsupNet):
    """Reconstruction variant of FITS (Xu et al., ICLR 2024): downsample by 2, rFFT, keep the low frequencies,
    a complex linear layer interpolates to the spectrum of length W, inverse FFT; weights shared across channels.
    Score = reconstruction error of the last 4 steps."""
    name = "FITS"
    cut = 6

    def _build(self, C):
        torch = _torch()
        nn = torch.nn
        Wl, Wo = W // 2, W // 2 + 1
        cut = min(self.cut, Wl // 2 + 1)  # W = 16 leaves only 5 bins after downsampling

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.re = nn.Linear(cut, Wo)
                s.im = nn.Linear(cut, Wo)

            def forward(s, x):  # x (B, W, C)
                xd = x[:, 1::2, :]                       # (B, W/2, C)
                mu = xd.mean(1, keepdim=True)
                f = torch.fft.rfft(xd - mu, dim=1)[:, :cut, :].transpose(1, 2)  # (B, C, cut)
                re = s.re(f.real) - s.im(f.imag)
                im = s.re(f.imag) + s.im(f.real)
                y = torch.fft.irfft(torch.complex(re, im), n=W, dim=2) * (W / Wl)
                return y.transpose(1, 2) + mu
        return M()

    def _loss(self, xb, yb, ep):
        return ((self.net(xb) - xb) ** 2).mean()

    def _score_batch(self, xb):
        return ((self.net(xb) - xb)[:, -4:, :] ** 2).mean((1, 2))


# ---------------------------------------------------------------- compact foundation model (zero-shot)

class TSPulseZS(Method):
    """TSPulse (Ekambaram et al., ICLR 2026; 1.08 M parameters, `main` revision recommended for AD), zero-shot.

    Every H steps the forecasting head runs on the 512 samples before t and scores t..t+H-1 by the mean squared
    forecast error over channels. The granite-tsfm AD pipeline is not used because it standardises with the
    statistics of the scored series and its reconstruction modes use centred windows. H = 4 fits the Pi 5 budget.
    """
    name = "TSPulse"
    H, CTX = 4, 512

    def __init__(self, seed=0, device=None):
        torch = _torch()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None

    def _load(self, C):
        import logging
        logging.disable(logging.WARNING)
        from tsfm_public.models.tspulse import TSPulseForReconstruction
        self.net = TSPulseForReconstruction.from_pretrained(
            "ibm-granite/granite-timeseries-tspulse-r1", revision="main", num_input_channels=C,
            mask_type="user").eval().to(self.dev)
        logging.disable(logging.NOTSET)

    def fit(self, tr, va):
        self._load(tr["X"][0].shape[1])
        return self

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))

    def score(self, X):
        torch = _torch()
        T, C = X.shape
        Xp = np.concatenate([np.repeat(X[:1], self.CTX, 0), X], 0).astype(np.float32)
        starts = np.arange(0, T, self.H)  # first scored step t; context = X[t-512 : t]
        out = np.zeros(T, np.float32)
        with torch.no_grad():
            for i in range(0, len(starts), 512):
                st = starts[i:i + 512]
                ctx = np.stack([Xp[t:t + self.CTX] for t in st])
                f = self.net(torch.from_numpy(ctx).to(self.dev))["forecast_output"][:, :self.H].float().cpu().numpy()
                for k, t in enumerate(st):
                    h = min(self.H, T - t)
                    out[t:t + h] = ((f[k, :h] - X[t:t + h]) ** 2).mean(1)
        return out


REGISTRY = {c.name: c for c in (MovingVar, SqDiff, PCAAD, IForestAD, HBOSAD, USAD, FITS, TSPulseZS,
                                LightGBMAD, ExtraTreesAD, CNN, TCN, LSTM)}


def timed(fn, *a):
    t = time.time()
    r = fn(*a)
    return r, time.time() - t
