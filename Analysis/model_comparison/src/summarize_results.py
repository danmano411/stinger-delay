"""Collect the 15 tuned-model results (plus baselines) into results/summary.csv and summary.md.

Baselines per dataset, scored on the same test split (the last two use TransLoc's ETA only, fit on train):
  - TransLoc as-is:   predict delay 0 (i.e. trust TransLoc's ETA)
  - Train mean:       predict the mean training delay
  - Line in ETA:      delay = a + b * eta_s
  - ETA-decile table: mean training delay within each decile of eta_s

summary.md opens with a quick scan: baselines vs the best tuned model, and error by ETA horizon.
"""
import json

import numpy as np
import pandas as pd

from mc_common import PREDS, RESULTS, load_model_ready, metrics

DATASETS = ["bus", "route", "combined"]
BASELINES = {  # name in summary.csv -> quick-scan label
    "TransLoc as-is (delay=0)": "TransLoc as-is (the current system)",
    "Train mean": "Add the average training delay to every ETA",
    "Line in ETA": "Straight-line fix: delay = a + b × ETA",
    "ETA-decile table": "Lookup table: average training delay per tenth of the ETA range",
}
BINS = [0, 120, 300, 600, 1200, np.inf]
HORIZONS = ["under 2 min", "2–5 min", "5–10 min", "10–20 min", "20+ min"]


def baseline_preds(d):
    et, ee, yt = d.X_train.eta_s.values, d.X_test.eta_s.values, d.y_train.values
    b, a = np.polyfit(et, yt, 1)
    edges = np.quantile(et, np.linspace(0, 1, 11))[1:-1]
    table = pd.Series(yt).groupby(np.digitize(et, edges)).mean()
    return dict(zip(BASELINES, [np.zeros(len(ee)), np.full(len(ee), yt.mean()), a + b * ee,
                                table.reindex(np.digitize(ee, edges)).fillna(yt.mean()).values]))


def span(vals, fmt, sep="–"):
    lo, hi = fmt(min(vals)), fmt(max(vals))
    return lo if lo == hi else f"{lo}{sep}{hi}"


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
    best, hz, within = {}, [], {}
    for ds in DATASETS:
        d = load_model_ready(ds)
        for name, pred in baseline_preds(d).items():
            m = metrics(d.y_test, pred)
            rows.append({"dataset": ds, "family": "baseline", "model": name, "r2": m["r2"], "mae_s": m["mae"],
                         "rmse_s": m["rmse"], "n_test": len(d.y_test), "tuning_mae_s": None})
        # horizon breakdown for the tuned model with the lowest test MAE
        best[ds] = min((r for r in rows if r["dataset"] == ds and r["family"] != "baseline"), key=lambda r: r["mae_s"])
        p = pd.read_csv(PREDS / f"{ds}__{best[ds]['family']}__{best[ds]['model']}.csv.gz")
        y = d.y_test.values
        assert np.allclose(p.y_true.values, y), f"{ds}: prediction file does not match the test split"
        err_tl, err_m = np.abs(y), np.abs(y - p.y_pred.values)
        within[ds] = ((err_tl < 120).mean(), (err_m < 120).mean())
        h = pd.cut(d.X_test.eta_s.values, BINS, labels=HORIZONS, right=False)
        g = pd.DataFrame({"horizon": h, "tl": err_tl, "model": err_m, "late": y}).groupby("horizon", observed=True).mean()
        hz.append(g.assign(dataset=ds))
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df.dataset, DATASETS, ordered=True)
    df["is_baseline"] = df.family == "baseline"
    df = df.sort_values(["dataset", "is_baseline", "r2"], ascending=[True, True, False]).drop(columns="is_baseline")
    df = df.reset_index(drop=True)
    df.to_csv(RESULTS / "summary.csv", index=False)

    # quick scan
    mae = df[df.family == "baseline"].pivot(index="model", columns="dataset", values="mae_s")
    lines = ["# Model comparison results", "", "## Quick scan", "",
             "Test = held-out service days (Mar 4, Mar 9). Error = mean absolute error of the predicted arrival time, "
             "in seconds. Predicting `delay_s` is the same as correcting TransLoc's ETA, so every row is comparable.", "",
             "| Average error (MAE) | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    for name, label in BASELINES.items():
        lines.append(f"| {label} | " + " | ".join(f"{mae.loc[name, ds]:.0f} s" for ds in DATASETS) + " |")
    lines.append("| **Best tuned model** (lowest test MAE) | "
                 + " | ".join(f"**{best[ds]['mae_s']:.0f} s** {best[ds]['model']}" for ds in DATASETS) + " |")
    simple = {ds: mae.loc[["Line in ETA", "ETA-decile table"], ds].min() for ds in DATASETS}
    tl = {ds: mae.loc["TransLoc as-is (delay=0)", ds] for ds in DATASETS}
    pct = "{:.0%}".format
    lines += ["", "The ETA fixes and lookup tables are fit on train, like the models. The better of the two gets "
              + span([(tl[ds] - simple[ds]) / (tl[ds] - best[ds]["mae_s"]) for ds in DATASETS], pct)
              + " of the best model's improvement over TransLoc. The best model then cuts the remaining error by another "
              + span([(simple[ds] - best[ds]["mae_s"]) / simple[ds] for ds in DATASETS], pct) + ".", "",
              "Error by how far away TransLoc says the bus is (ranges across the datasets):", "",
              "| TransLoc's ETA | TransLoc's error | Best model's error | Bus arrives later than TransLoc said, on average |",
              "|---|---|---|---|"]
    hz = pd.concat(hz)
    for h in HORIZONS:
        g = hz.loc[hz.index == h]
        if g.empty:
            continue
        only = f" ({', '.join(g.dataset)} only)" if len(g) < len(DATASETS) else ""
        lines.append(f"| {h}{only} | {span(g.tl, '{:.0f}'.format)} s | {span(g.model, '{:.0f}'.format)} s | "
                     f"{span(g.late / 60, '{:+.1f}'.format, ' to ')} min |")
    lines += ["", f"Within 2 min of the actual arrival: TransLoc {span([w[0] for w in within.values()], pct)} "
              f"of predictions, best model {span([w[1] for w in within.values()], pct)}.", ""]

    # full table
    tuned = df[df.family != "baseline"]
    lines += ["## All models and baselines", "",
              f"{len(tuned)} tuned models. Target `delay_s` (seconds the bus arrived later than TransLoc's ETA); "
              "test split = held-out service days (Mar 4, Mar 9), scored once per model.", "",
              "| Dataset | Family | Model | Test R² | Test MAE (s) | Test RMSE (s) | n_test |", "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.dataset} | {r.family} | {r.model} | {r.r2:.3f} | {r.mae_s:.1f} | {r.rmse_s:.1f} | {r.n_test:,} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
