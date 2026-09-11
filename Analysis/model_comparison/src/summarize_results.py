"""Collect the 15 tuned-model results (plus baselines) into results/summary.csv and summary.md.

Baselines per dataset, scored on the same test split:
  - TransLoc as-is: predict delay 0 (i.e. trust TransLoc's ETA)
  - Train mean:     predict the mean training delay
"""
import json

import numpy as np
import pandas as pd

from mc_common import RESULTS, load_model_ready, metrics

DATASETS = ["bus", "route", "combined"]


def main():
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("smoke"):
            continue
        ex = r.get("extra") or {}
        # tuning score: grouped-CV MAE (lazy) or best inner-holdout MAE (xgboost / nn); optimistic vs test
        tuning = r.get("cv_mae") or ex.get("best_trial_val_mae") or ex.get("ensemble_val_mae")
        rows.append({"dataset": r["dataset"], "family": r["family"], "model": r["model"], "r2": r["r2"],
                     "mae_s": r["mae"], "rmse_s": r["rmse"], "n_test": r["n_test"], "tuning_mae_s": tuning})
    for ds in DATASETS:
        d = load_model_ready(ds)
        for name, pred in [("TransLoc as-is (delay=0)", np.zeros(len(d.y_test))),
                           ("Train mean", np.full(len(d.y_test), d.y_train.mean()))]:
            m = metrics(d.y_test, pred)
            rows.append({"dataset": ds, "family": "baseline", "model": name, "r2": m["r2"], "mae_s": m["mae"],
                         "rmse_s": m["rmse"], "n_test": len(d.y_test), "tuning_mae_s": None})
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df.dataset, DATASETS, ordered=True)
    df["is_baseline"] = df.family == "baseline"
    df = df.sort_values(["dataset", "is_baseline", "r2"], ascending=[True, True, False]).drop(columns="is_baseline")
    df = df.reset_index(drop=True)
    df.to_csv(RESULTS / "summary.csv", index=False)
    tuned = df[df.family != "baseline"]
    lines = ["# Model comparison results", "",
             f"{len(tuned)} tuned models. Target `delay_s` (seconds the bus arrived later than TransLoc's ETA); "
             "test split = held-out service days (Mar 4, Mar 9), scored once per model.", "",
             "| Dataset | Family | Model | Test R² | Test MAE (s) | Test RMSE (s) | n_test |", "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.dataset} | {r.family} | {r.model} | {r.r2:.3f} | {r.mae_s:.1f} | {r.rmse_s:.1f} | {r.n_test:,} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
