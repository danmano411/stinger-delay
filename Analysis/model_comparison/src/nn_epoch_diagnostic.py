"""Diagnostic (train split only, test never touched): what happens to the tuned NN past its early stop?
Run from src/: python nn_epoch_diagnostic.py <bus|route|combined> <threads>  -> results/logs/nn_epoch_diagnostic_<ds>.json

A  as run:        the final fit's schedule (OneCycle sized for 300 epochs, or plateau), val = the held-out train day
B  schedule fix:  OneCycle sized for the 60 epochs actually trained, same val day
C  no day shift:  same as B, but val = held-out 3-hour blocks of the training day(s) instead of another day
No early stopping: every run trains 60 epochs and logs train MAE, val MAE and LR each epoch.
"""
import json, sys, time

import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import mc_common as mc

ds, EPOCHS, SEEDS = sys.argv[1], 60, [42, 43, 44]
torch.set_num_threads(int(sys.argv[2]))
cfg = json.loads((mc.RESULTS / f"{ds}__nn__MLP.json").read_text())["params"]
d = mc.load_model_ready(ds)
Xa, Xv, ya, yv, ga = mc.group_val_split(d.X_train, d.y_train, d.g_train, frac=0.2, seed=42)  # same split as the NN scripts


class MLP(nn.Module):
    def __init__(self, n, width, depth, dropout):
        super().__init__()
        layers, k = [], n
        for _ in range(depth):
            layers += [nn.Linear(k, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout)]
            k = width
        self.net = nn.Sequential(*layers, nn.Linear(k, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run(Xtr, ytr, Xva, yva, sched_epochs, seed):
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr).astype(np.float32)
    V = np.clip(sc.transform(Xva), A.min(0), A.max(0)).astype(np.float32)
    mu, sd = float(ytr.mean()), float(ytr.std())
    ya_ = ((ytr.to_numpy() - mu) / sd).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(A.shape[1], cfg["width"], cfg["depth"], cfg["dropout"])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    dl = DataLoader(TensorDataset(torch.from_numpy(A), torch.from_numpy(ya_)), batch_size=cfg["batch_size"], shuffle=True)
    onecycle = cfg["scheduler"] == "onecycle"
    sch = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=sched_epochs * len(dl)) if onecycle
           else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3))
    loss_fn, At, Vt = nn.SmoothL1Loss(beta=cfg["beta"]), torch.from_numpy(A), torch.from_numpy(V)
    tr_h, va_h, lr_h = [], [], []
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(m(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if onecycle:
                sch.step()
        m.eval()
        with torch.no_grad():
            tr_h.append(float(np.abs(m(At).numpy() * sd + mu - ytr.to_numpy()).mean()))
            va_h.append(float(np.abs(m(Vt).numpy() * sd + mu - yva.to_numpy()).mean()))
        lr_h.append(opt.param_groups[0]["lr"])
        if not onecycle:
            sch.step(va_h[-1])
    return {"train": tr_h, "val": va_h, "lr": lr_h}


def baselines(Xtr, ytr, Xva, yva):
    b, a = np.polyfit(Xtr.eta_s, ytr, 1)
    return {"const_val_mae": float(np.abs(yva - ytr.mean()).mean()),
            "line_val_mae": float(np.abs(yva - (a + b * Xva.eta_s)).mean())}


# C: hold out 3-hour blocks of the inner-train day(s)
blk = pd.Series(ga).str[:10].to_numpy() + "_" + (Xa.hour.to_numpy() // 3).astype(int).astype(str)
ci, cv = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(Xa, ya, blk))
exps = {"A": (Xa, ya, Xv, yv, 300)}
if cfg["scheduler"] == "onecycle":
    exps["B"] = (Xa, ya, Xv, yv, EPOCHS)
    exps["C"] = (Xa.iloc[ci], ya.iloc[ci], Xa.iloc[cv], ya.iloc[cv], EPOCHS)
out = {"cfg": cfg, "val_groups": sorted(set(np.asarray(d.g_train)) - set(ga)), "n": {}, "baselines": {}, "runs": {}}
for k, (Xtr, ytr, Xva, yva, se) in exps.items():
    out["n"][k] = [len(Xtr), len(Xva)]
    out["baselines"][k] = baselines(Xtr, ytr, Xva, yva)
    for s in SEEDS:
        t = time.time()
        out["runs"][f"{k}_{s}"] = run(Xtr, ytr, Xva, yva, se, s)
        print(ds, k, s, f"{time.time() - t:.0f}s", flush=True)
p = mc.LOGS / f"nn_epoch_diagnostic_{ds}.json"
p.write_text(json.dumps(out))
print("saved", p)
