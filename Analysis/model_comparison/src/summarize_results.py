"""Build results/summary.csv and results/summary.md from the saved results and test predictions.

summary.md, in reading order:
  1. Best model per dataset vs TransLoc, and how much of the gain a lookup table on TransLoc's ETA already gets
  2. Error by how far away TransLoc says the bus is
  3. Share of predictions within 2 min of the actual arrival
  4. By route, and pooled vs single-route models
  5. Next steps
  6. All 15 tuned models plus 4 baselines

Baselines, scored on the same test split (the last two use only TransLoc's ETA and are fit on train):
TransLoc as-is (predict delay 0), train mean, a straight line in the ETA, and a lookup table by ETA decile.
Percentages are computed from the rounded numbers shown, so every calculation in summary.md checks by hand.
"""
import json

import numpy as np
import pandas as pd

from mc_common import PREDS, RESULTS, load_features, load_model_ready, metrics

DATASETS = ["bus", "route", "combined"]
BASELINES = ["TransLoc as-is (delay=0)", "Train mean", "Straight line in ETA", "Lookup table (ETA deciles)"]
BINS = [0, 120, 300, 600, 1200, np.inf]
HORIZONS = ["under 2 min", "2–5 min", "5–10 min", "10–20 min", "20+ min"]
KEY = ["t_local", "vehicle_id", "route_stop_id"]  # identifies a test row across datasets


def baseline_preds(d):
    et, ee, yt = d.X_train.eta_s.values, d.X_test.eta_s.values, d.y_train.values
    b, a = np.polyfit(et, yt, 1)
    edges = np.quantile(et, np.linspace(0, 1, 11))[1:-1]
    table = pd.Series(yt).groupby(np.digitize(et, edges)).mean()
    return dict(zip(BASELINES, [np.zeros(len(ee)), np.full(len(ee), yt.mean()), a + b * ee,
                                table.reindex(np.digitize(ee, edges)).fillna(yt.mean()).values]))


def mae(t, col="pred"):
    return round(float(np.abs(t.y - t[col]).mean()))


def calc(a, b, base):
    return f"({a} − {b}) / {base} = {100 * (a - b) / base:.0f}%"


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


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
    best = {}
    for ds in DATASETS:
        d = load_model_ready(ds)
        preds = baseline_preds(d)
        for name, pred in preds.items():
            m = metrics(d.y_test, pred)
            rows.append({"dataset": ds, "family": "baseline", "model": name, "r2": m["r2"], "mae_s": m["mae"],
                         "rmse_s": m["rmse"], "n_test": len(d.y_test), "tuning_mae_s": None})
        b = dict(min((r for r in rows if r["dataset"] == ds and r["family"] != "baseline"), key=lambda r: r["mae_s"]))
        p = pd.read_csv(PREDS / f"{ds}__{b['family']}__{b['model']}.csv.gz")
        assert np.allclose(p.y_true.values, d.y_test.values), f"{ds}: prediction file does not match the test split"
        b["test"] = d.meta_test[KEY + ["route_name"]].reset_index(drop=True).assign(
            y=d.y_test.values, pred=p.y_pred.values, zero=0.0, lookup=preds[BASELINES[3]],
            horizon=pd.cut(d.X_test.eta_s.values, BINS, labels=HORIZONS, right=False))
        best[ds] = b
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df.dataset, DATASETS, ordered=True)
    df["is_baseline"] = df.family == "baseline"
    df = df.sort_values(["dataset", "is_baseline", "mae_s"]).drop(columns="is_baseline")
    df = df.reset_index(drop=True)
    df.to_csv(RESULTS / "summary.csv", index=False)

    T = {ds: mae(best[ds]["test"], "zero") for ds in DATASETS}
    L = {ds: mae(best[ds]["test"], "lookup") for ds in DATASETS}
    M = {ds: mae(best[ds]["test"]) for ds in DATASETS}
    lines = [
        "# Results", "",
        "Test days are held out whole (Mar 4 and Mar 9), and each model is scored on them once. Error is the mean "
        "absolute error (MAE) of the predicted arrival time, in seconds. Predicting `delay_s` is the same as "
        "correcting TransLoc's ETA, so \"TransLoc as-is\" is simply a prediction of zero delay.", "",
        "**Best model** means the tuned model with the lowest test MAE on that dataset: "
        + ", ".join(f"{best[ds]['model']} for `{ds}`" for ds in DATASETS)
        + ". It is picked on the test days, so its score is slightly optimistic. "
        "Percentages are computed from the rounded numbers shown.", "",
        "## 1. Best model vs TransLoc, and where the gain comes from", "",
        "The lookup table is the simplest possible fix: it uses nothing but TransLoc's ETA, splits it into 10 ranges, "
        "and adds the average training delay for that range. The best model adds the rest using the other features.", "",
        row("", *[f"`{ds}`" for ds in DATASETS]), row(*["---"] * 4),
        row("Best model", *[best[ds]["model"] for ds in DATASETS]),
        row("**T** TransLoc as-is", *[f"{T[ds]} s" for ds in DATASETS]),
        row("**L** Lookup table on TransLoc's ETA", *[f"{L[ds]} s" for ds in DATASETS]),
        row("**M** Best model", *[f"{M[ds]} s" for ds in DATASETS]),
        row("**Error reduction** (T − M) / T", *[f"**{calc(T[ds], M[ds], T[ds])}**" for ds in DATASETS]),
        row("… from the lookup table (T − L) / T", *[calc(T[ds], L[ds], T[ds]) for ds in DATASETS]),
        row("… added by the model (L − M) / T", *[calc(L[ds], M[ds], T[ds]) for ds in DATASETS]),
        "", "Two more baselines (the training mean delay, and a straight line in the ETA) are in section 6.", "",
        "## 2. Error by how far away TransLoc says the bus is", "",
        row("Dataset (best model)", "TransLoc's ETA", "Rows", "TransLoc error", "Model error",
            "Reduction (TransLoc − model) / TransLoc"), row(*["---"] * 6)]
    within, late = [], {}
    for ds in DATASETS:
        t = best[ds]["test"]
        groups = [(h, g) for h, g in t.groupby("horizon", observed=True)] + [("**all**", t)]
        for i, (h, g) in enumerate(groups):
            name = f"`{ds}` ({best[ds]['model']})" if i == 0 else ""
            a, m = mae(g, "zero"), mae(g)
            lines.append(row(name, h, f"{len(g):,}", f"{a} s", f"{m} s", calc(a, m, a)))
            tl, mm = round(100 * (g.y.abs() < 120).mean()), round(100 * ((g.y - g.pred).abs() < 120).mean())
            within.append(row(name, h, f"{tl}%", f"{mm}%", f"{mm} − {tl} = {mm - tl:+d} pts"))
            late.setdefault(h, []).append(g.y.mean() / 60)
    span = lambda v: f"{min(v):.1f}" if f"{min(v):.1f}" == f"{max(v):.1f}" else f"{min(v):.1f} to {max(v):.1f}"
    lines += ["", f"TransLoc's error is almost all lateness. On average the bus arrives {span(late[HORIZONS[0]])} min "
              f"later than TransLoc says when it is under 2 min away, and {span(late['10–20 min'])} min later when it "
              "is 10–20 min away. That steady, growing optimism is what the lookup table in section 1 corrects.", "",
              "## 3. How often the prediction is within 2 minutes of the actual arrival", "",
              row("Dataset (best model)", "TransLoc's ETA", "TransLoc", "Model", "Gain"), row(*["---"] * 5), *within]

    # 4. by route (combined) and pooled vs single-route models on identical test rows
    c = best["combined"]["test"]
    per_route = [f"{r} {mae(g, 'zero')} → {mae(g)} s ({100 * (mae(g, 'zero') - mae(g)) / mae(g, 'zero'):.0f}%)"
                 for r, g in c.groupby("route_name")]
    same = {ds: best[ds]["test"][KEY + ["y", "pred"]].merge(c[KEY + ["pred"]], on=KEY, suffixes=("", "_c"))
            for ds in ["route", "bus"]}
    assert all(len(same[ds]) == len(best[ds]["test"]) for ds in same), "route/bus test rows missing from combined"
    lines += ["", "## 4. By route, and pooled vs single-route models", "",
              f"- **The gain is uneven by route.** On the `combined` test days, the best model's error by route is "
              f"{', '.join(per_route)}. Red's feed only gives next-stop ETAs, so TransLoc is already close there.",
              "- **Pooling routes helped, even for a single bus.** On identical test rows, the `combined` model beat "
              f"the models trained on one route or one bus: on Green's test day {mae(same['route'], 'pred_c')} s vs "
              f"{mae(same['route'])} s for the best `route` model, and on bus #3's rows "
              f"{mae(same['bus'], 'pred_c')} s vs {mae(same['bus'])} s for the best `bus` model. "
              "`combined` has five training days instead of one or two. This rests on one test day, so treat it as a first sign."]

    # 5. next steps
    f, s = load_features("combined")
    arrival = (f.t_local + pd.to_timedelta(f.eta_s + f[s["target"]], unit="s")).dt.round("10s")
    n_arr = pd.DataFrame({"v": f.vehicle_id, "s": f.route_stop_id, "a": arrival}).drop_duplicates().shape[0]
    day = f.groupby(["route_name", "date"])[s["target"]].mean()
    swing = day.groupby(level=0).agg(lambda v: v.max() - v.min()).idxmax()
    lo, hi = day[swing].idxmin(), day[swing].idxmax()
    md = lambda x: f"{pd.Timestamp(x):%b} {pd.Timestamp(x).day}"
    lines +=["", "## 5. Next steps", "",
              f"1. **Collect more days, with all routes recorded on the same days.** The {len(f):,} labeled rows "
              f"are only about {round(n_arr, -2):,} distinct bus-stop arrivals over {f.date.nunique()} days, and each "
              f"route has two training days. Days differ a lot ({swing}'s average delay was {day[swing, lo]:.0f} s on "
              f"{md(lo)} and {day[swing, hi]:.0f} s on {md(hi)}), and what a model learns from one day mostly doesn't carry "
              "to the next (README section 6). More days also tighten the error bars: with one test day "
              "per route there is no honest interval yet, and it narrows with the square root of the number of days.",
              "2. **Ask GT Parking & Transportation for historical data.** The public API has no history, but the "
              "RideSystems dispatch side may keep it. Months of data at once would beat a semester of scraping.",
              "3. **Fix the scrapers and save more fields.** Recover Clough's stop ID and ETAs of 0 s (saved as blank), "
              "and start saving occupancy, IsDelayed, OnTimeStatus and heading.",
              "4. **Keep the pooled model, and test per-route models as data grows.** With sparse data it seems natural "
              "to fit each route on its own, but on Green pooling already won (section 4). Gold and Red never got "
              "their own models, so train per-route and pooled models on the same days and compare them on the same "
              "rows. A pooled model with route and stop embeddings is also the one that can learn what routes share, "
              "such as the campus road layout, traffic and weather.",
              "5. **Evaluate day by day.** Train on everything before a day, test on that day, and repeat. This gives "
              "a per-day error distribution with real confidence intervals, by route and by distance.",
              "6. **Ship the lookup table now.** It uses only TransLoc's ETA, is easy to explain, and gets most of the "
              "gain (section 1). Refit it as data comes in, keep gradient boosting as the upgrade, and shelve the NN "
              "until there are dozens of days.",
              "7. **Add features that explain bad days:** GT class-change times, game days and other events, "
              "occupancy, and the gap to the bus ahead."]

    # 6. every model
    tuned = df[df.family != "baseline"]
    lines += ["", "## 6. All models and baselines", "",
              f"{len(tuned)} tuned models (5 per dataset) and 4 baselines per dataset, sorted by test MAE, so each "
              "dataset's first row is its best model. Tuning MAE and every number here are also in `summary.csv`.", "",
              "| Dataset | Family | Model | Test R² | Test MAE (s) | Test RMSE (s) | n_test |", "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.dataset} | {r.family} | {r.model} | {r.r2:.3f} | {r.mae_s:.1f} | {r.rmse_s:.1f} | {r.n_test:,} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
