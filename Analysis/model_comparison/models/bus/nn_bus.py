"""T4.bus.nn — tuned PyTorch MLP for the `bus` dataset (delay_s regression).

Run from this folder:
    MC_SMOKE=1 MC_THREADS=2 python nn_bus.py        (smoke, <5 min)
    MC_THREADS=2 python nn_bus.py                    (full run, T5 launches this)

Pipeline: load model-ready data -> inner train/val split (group-disjoint) ->
scale X (StandardScaler) and standardise y, both fit on inner-train only ->
Optuna-tune an MLP (width/depth/dropout/loss-beta/lr/wd/batch/scheduler)
against inner-val MAE with early stopping and pruning -> refit the best
config -> predict the test split once -> mc_common.save_result.
"""
import sys
from pathlib import Path

# This file lives at models/bus/nn_bus.py -> parents[2] is model_comparison/, per protocol.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")  # headless: no display available when run as a script
import matplotlib.pyplot as plt

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mc_common import (SMOKE, THREADS, budget, set_seed, load_model_ready,
                        group_val_split, metrics, save_result, PLOTS)

optuna.logging.set_verbosity(optuna.logging.WARNING)
set_seed(42)
torch.set_num_threads(THREADS)

DS = "bus"

# --------------------------------------------------------------- N1: data
d = load_model_ready(DS)
# Inner group-disjoint split of TRAIN only (test is never touched here).
X_tr_df, X_val_df, y_tr_s, y_val_s, g_tr = group_val_split(d.X_train, d.y_train, d.g_train, frac=0.2, seed=42)

# Scale X on inner-train only; target standardised with inner-train mean/std.
scaler = StandardScaler().fit(X_tr_df)
X_tr = scaler.transform(X_tr_df).astype(np.float32)
X_val = scaler.transform(X_val_df).astype(np.float32)
X_test = scaler.transform(d.X_test).astype(np.float32)
# Clip val/test inputs to the inner-train range: an MLP extrapolates linearly, and whole held-out days
# bring feature values never seen in training (e.g. humidity). Trees are bounded by construction.
x_lo, x_hi = X_tr.min(axis=0), X_tr.max(axis=0)
X_val, X_test = np.clip(X_val, x_lo, x_hi), np.clip(X_test, x_lo, x_hi)

y_mean, y_std = float(y_tr_s.mean()), float(y_tr_s.std())
y_tr = ((y_tr_s.to_numpy() - y_mean) / y_std).astype(np.float32)
y_val_orig = y_val_s.to_numpy(dtype=np.float32)  # kept in seconds for MAE-based early stopping
y_test_orig = d.y_test.to_numpy(dtype=np.float64)  # exact targets in the saved preds

X_val_t = torch.from_numpy(X_val)
X_test_t = torch.from_numpy(X_test)
n_features = X_tr.shape[1]
print(f"[N1] X_tr {X_tr.shape}  X_val {X_val.shape}  X_test {X_test.shape}  n_features={n_features}")


# --------------------------------------------------------------- N2: model
class MLP(nn.Module):
    """depth x [Linear -> LayerNorm -> SiLU -> Dropout] -> Linear(1)."""

    def __init__(self, n_features, width=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = n_features
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------- N3: training loop
def train_model(cfg, max_epochs, trial=None):
    """Train one MLP config with early stopping on inner-val MAE (seconds).
    Returns (model_with_best_weights, history[list of val MAE per epoch], best_val_mae, epochs_trained)."""
    model = MLP(n_features, cfg["width"], cfg["depth"], cfg["dropout"])
    loss_fn = nn.SmoothL1Loss(beta=cfg["beta"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                         batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    if cfg["scheduler"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"],
                                                      total_steps=max_epochs * len(loader))
        step_per_batch = True
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        step_per_batch = False

    history, best_val_mae, best_state, patience, epochs_trained = [], float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step_per_batch:
                sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).numpy() * y_std + y_mean  # inverse-transform to seconds
        val_mae = float(np.mean(np.abs(val_pred - y_val_orig)))
        history.append(val_mae)
        epochs_trained = epoch + 1
        if not step_per_batch:
            sched.step(val_mae)

        if trial is not None:
            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_mae < best_val_mae - 1e-6:
            best_val_mae, patience = val_mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:  # early stopping patience
                break

    model.load_state_dict(best_state)  # restore best weights
    return model, history, best_val_mae, epochs_trained


# --------------------------------------------------------------- N4: Optuna tuning
def objective(trial):
    cfg = {
        "width": trial.suggest_categorical("width", [64, 128, 256, 512]),
        "depth": trial.suggest_int("depth", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "beta": trial.suggest_categorical("beta", [0.5, 1.0]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "scheduler": trial.suggest_categorical("scheduler", ["onecycle", "plateau"]),
    }
    _, _, best_val_mae, _ = train_model(cfg, max_epochs=budget(150, 4), trial=trial)
    return best_val_mae


study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42),
                             pruner=MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=budget(30, 2), timeout=budget(1800, 90))
print(f"[N4] best trial value={study.best_value:.2f}s  params={study.best_params}")


# --------------------------------------------------------------- N5: final fit + test
# One re-initialisation of a small MLP varies a lot on this data (a single retrain of the best
# config landed 10-25 s of val MAE away from the tuned trial), so the final model averages the
# best config trained from several seeds. Each member early-stops on inner-val; test is used once.
n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
print(f"[N4] trials: {n_complete} complete, {n_pruned} pruned")
members = []
for k in range(budget(5, 1)):
    set_seed(42 + k)
    members.append(train_model(study.best_params, max_epochs=budget(300, 6), trial=None))

with torch.no_grad():
    for m, *_ in members:
        m.eval()
    val_pred = np.mean([m(X_val_t).numpy() for m, *_ in members], axis=0) * y_std + y_mean
    y_pred = np.mean([m(X_test_t).numpy() for m, *_ in members], axis=0).astype(np.float64) * y_std + y_mean
ensemble_val_mae = float(np.mean(np.abs(val_pred - y_val_orig)))
print(f"[N5] {len(members)}-seed ensemble inner-val MAE {ensemble_val_mae:.2f}s "
      f"(members: {[round(v, 1) for _, _, v, _ in members]})")

save_result(DS, "nn", "MLP", y_test_orig, y_pred, study.best_params, cv_mae=ensemble_val_mae,
            extra={"n_seeds": len(members), "ensemble_val_mae": ensemble_val_mae,
                   "member_val_mae": [v for _, _, v, _ in members],
                   "epochs_trained": [e for _, _, _, e in members],
                   "n_trials_complete": n_complete, "n_trials_pruned": n_pruned,
                   "best_trial_val_mae": study.best_value,
                   "history": [h for _, h, _, _ in members]})


# --------------------------------------------------------------- N6: artifacts
PLOTS.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(6, 4))
for k, (_, h, _, _) in enumerate(members):
    plt.plot(range(1, len(h) + 1), h, label=f"seed {42 + k}")
plt.xlabel("epoch")
plt.ylabel("inner-val MAE (s)")
plt.title(f"{DS} NN learning curves (final fit)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(PLOTS / f"{DS}__nn_curve.png", dpi=100)
plt.close()

test_metrics = metrics(y_test_orig, y_pred)
print("[N6] final metrics table:")
print(f"  {'dataset':<6} {'family':<3} {'model':<4} {'r2':>8} {'mae':>8} {'rmse':>8}")
print(f"  {DS:<6} {'nn':<3} {'MLP':<4} {test_metrics['r2']:>8.4f} {test_metrics['mae']:>8.1f} {test_metrics['rmse']:>8.1f}")
