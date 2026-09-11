"""Shared helpers for the model comparison: paths, run budgets, data loading, metrics,
result saving and the EDA pruning rules. Every notebook/script imports from here so all
15 models use the same split, the same features and the same metric code."""
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

MC = Path(__file__).resolve().parents[1]
DATASETS = MC / "datasets"
RESULTS = MC / "results"
PREDS, PLOTS, LOGS = RESULTS / "preds", RESULTS / "plots", RESULTS / "logs"

SMOKE = os.environ.get("MC_SMOKE") == "1"          # tiny budgets for a quick end-to-end check
THREADS = int(os.environ.get("MC_THREADS", "2"))   # per-job CPU cap (the machine runs 9 jobs at once)


def budget(full, smoke):
    """Pick the full-run value, or the smoke-test value when MC_SMOKE=1."""
    return smoke if SMOKE else full


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _read(stem):
    schema = json.loads((DATASETS / f"{stem}_schema.json").read_text(encoding="utf-8"))
    return pd.read_parquet(DATASETS / f"{stem}.parquet"), schema


def load_features(ds):
    """Output of build_datasets.py -> (DataFrame, schema dict)."""
    return _read(f"{ds}_features")


def save_model_ready(ds, df, features, notes=None):
    """Write the EDA-filtered dataset; the schema keeps meta/target/split info from the build."""
    _, schema = load_features(ds)
    target = schema["target"]
    out = df[schema["meta"] + list(features) + [target]].reset_index(drop=True)
    assert not out[list(features) + [target]].isna().any().any(), "nulls in model-ready data"
    const = [c for c in features if out[c].nunique() <= 1]
    assert not const, f"constant features: {const}"
    out.to_parquet(DATASETS / f"{ds}_model_ready.parquet", index=False)
    new = {**schema, "features": list(features), "n_rows": len(out), "eda_notes": notes or {}}
    (DATASETS / f"{ds}_model_ready_schema.json").write_text(json.dumps(new, indent=2, default=_jsonable), encoding="utf-8")
    return out


def load_model_ready(ds):
    """-> namespace with X_train, X_test, y_train, y_test (pandas), g_train (cv groups), features, target."""
    df, s = _read(f"{ds}_model_ready")
    tr = df[df[s["split_col"]] == "train"].reset_index(drop=True)
    te = df[df[s["split_col"]] == "test"].reset_index(drop=True)
    f, y = s["features"], s["target"]
    return SimpleNamespace(df=df, schema=s, features=f, target=y,
                           X_train=tr[f], X_test=te[f], y_train=tr[y], y_test=te[y],
                           g_train=tr[s["group_col"]].to_numpy(), meta_test=te[s["meta"]])


def group_val_split(X, y, groups, frac=0.2, seed=42):
    """Group-disjoint inner split of the TRAIN set (for screening / early stopping)."""
    groups = np.asarray(groups)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed).split(X, y, groups))
    pick = lambda a, i: a.iloc[i] if hasattr(a, "iloc") else a[i]
    return pick(X, tr), pick(X, va), pick(y, tr), pick(y, va), groups[tr]


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (np.ndarray, pd.Series)):
        return o.tolist()
    return str(o)


def save_result(ds, family, model_name, y_test, y_pred, params, cv_mae=None, extra=None):
    """Score the final model on the test split and persist metrics + predictions."""
    fam = family + ("_smoke" if SMOKE else "")
    rid = f"{ds}__{fam}__{model_name}"
    rec = {"id": rid, "dataset": ds, "family": fam, "model": model_name, **metrics(y_test, y_pred),
           "n_test": int(len(y_test)), "params": params, "cv_mae": cv_mae, "smoke": SMOKE,
           "threads": THREADS, "extra": extra or {}}
    PREDS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": np.asarray(y_test, float), "y_pred": np.asarray(y_pred, float)}).to_csv(
        PREDS / f"{rid}.csv.gz", index=False)
    (RESULTS / f"{rid}.json").write_text(json.dumps(rec, indent=2, default=_jsonable), encoding="utf-8")
    print(f"[{rid}] test R2={rec['r2']:.4f}  MAE={rec['mae']:.1f}s  RMSE={rec['rmse']:.1f}s  (n={rec['n_test']})")
    return rec


# ---------------------------------------------------------------- EDA rules
def iqr_fences(s, k=3.0):
    q1, q3 = np.quantile(np.asarray(s, float), [0.25, 0.75])
    return float(q1 - k * (q3 - q1)), float(q3 + k * (q3 - q1))


def high_corr_pairs(df, cols, method="spearman", thresh=0.95):
    c = df[list(cols)].corr(method=method)
    rows = [(a, b, c.loc[a, b]) for i, a in enumerate(c.columns) for b in c.columns[i + 1:] if abs(c.loc[a, b]) >= thresh]
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", method]).sort_values(method, key=abs, ascending=False)


def prune_correlated(train_df, cols, target, thresh=0.95, method="spearman", protect=("eta_s",)):
    """For every pair with |corr| >= thresh, drop the member less correlated with the target.
    Fit on TRAIN rows only. Returns (kept_features, dropped_records)."""
    cols = list(cols)
    corr = train_df[cols].corr(method=method).abs()
    tcorr = train_df[cols].corrwith(train_df[target], method=method).abs().fillna(0)
    pairs = sorted(((corr.loc[a, b], a, b) for i, a in enumerate(cols) for b in cols[i + 1:]
                    if corr.loc[a, b] >= thresh), reverse=True)
    keep, dropped = list(cols), []
    for r, a, b in pairs:
        if a not in keep or b not in keep:
            continue
        victim = b if a in protect else a if b in protect else (a if tcorr[a] < tcorr[b] else b)
        kept = b if victim == a else a
        keep.remove(victim)
        dropped.append({"dropped": victim, "kept": kept, method: round(float(r), 4),
                        "target_corr_dropped": round(float(tcorr[victim]), 4),
                        "target_corr_kept": round(float(tcorr[kept]), 4)})
    return keep, dropped
