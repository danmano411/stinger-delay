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


def lookup_table(d):
    """L: cut train rows into 10 equal-count ETA ranges; each range stores its mean training delay."""
    et = d.X_train.eta_s.values
    edges = np.quantile(et, np.linspace(0, 1, 11))[1:-1]
    return edges, pd.Series(d.y_train.values).groupby(np.digitize(et, edges)).mean()


def baseline_preds(d):
    et, ee, yt = d.X_train.eta_s.values, d.X_test.eta_s.values, d.y_train.values
    b, a = np.polyfit(et, yt, 1)
    edges, table = lookup_table(d)
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
    best, example = {}, None
    for ds in DATASETS:
        d = load_model_ready(ds)
        if ds == "route":  # worked example of L: the ETA range holding 7 min
            edges, table = lookup_table(d)
            i = int(np.digitize(420, edges))
            bounds = [d.X_train.eta_s.min(), *edges, d.X_train.eta_s.max()]
            example = (bounds[i] / 60, bounds[i + 1] / 60, table[i] / 60)
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
        "**L, the lookup table,** is a correction built for this study, not something TransLoc provides. It looks "
        "only at TransLoc's ETA. The training rows are sorted by ETA and cut into 10 ranges with the same number of "
        "rows each, and each range stores the average delay seen in training. To predict, find the range the ETA "
        "falls in and add that range's average delay. For example, on `route` one range is "
        f"{example[0]:.1f}–{example[1]:.1f} min with an average delay of {example[2]:+.1f} min, so when TransLoc says "
        f"7 min, L predicts {7 + example[2]:.1f} min. The best model (**M**) adds the rest of the gain using the "
        "other features.", "",
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
    within, late, red, gain = [], {}, {}, {}
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
            red.setdefault(h, []).append((ds, round(100 * (a - m) / a)))
            gain.setdefault(h, []).append(mm - tl)
    span = lambda v: f"{min(v):.1f}" if f"{min(v):.1f}" == f"{max(v):.1f}" else f"{min(v):.1f} to {max(v):.1f}"
    mid = ["2–5 min", "5–10 min", "10–20 min"]
    rng = lambda v: f"{min(v)}" if min(v) == max(v) else f"{min(v)}–{max(v)}"
    r_mid, g_mid = [x for h in mid for _, x in red[h]], [x for h in mid for x in gain[h]]
    far = red.get(HORIZONS[-1], [])
    lines += ["", "**The models help most from 2 to 20 minutes out**, which is likely when riders decide whether to head "
              f"to the stop (an assumption; there is no rider data here). In that range the error falls by {rng(r_mid)}% "
              f"and the share of predictions within 2 minutes of the actual arrival rises by {rng(g_mid)} points "
              f"(section 3). Under 2 minutes TransLoc is already fairly close ({rng([x for _, x in red[HORIZONS[0]]])}% cut)"
              + (f", and beyond 20 minutes ({', '.join(f'`{ds}`' for ds, _ in far)} only) the cut is "
                 f"{rng([x for _, x in far])}%." if far else "."),
              "", f"TransLoc's error is almost all lateness. On average the bus arrives {span(late[HORIZONS[0]])} min "
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
    lines += ["", "## 5. Next steps", "",
              "### 5.1 Model structure: `route` or `combined`, decided by the data", "",
              "- **Drop per-bus models.** A live system needs a prediction for every bus, one bus has the least data, "
              "and on bus #3's own rows the pooled model was better (section 4). Bus identity works better as a feature "
              "inside a larger model, as the bus embeddings already are.",
              "- **Choose between `route` and `combined` from the data we collect.** Today `combined` wins on Green "
              "(section 4), because pooling adds days no single route has. If each route gets many days of its own, "
              "per-route models may win where routes behave differently (Red's feed, for instance, only gives "
              "next-stop ETAs). Decide per route: train both on the same days, evaluate day by day (train on everything "
              "before a day, test on that day, repeat), and keep whichever wins. A middle path is one pooled model with "
              "a per-route correction on top.", "",
              "### 5.2 More information to collect", "",
              row("Source", "What it adds", "Notes"), row(*["---"] * 3),
              row("More days, all routes on the same days",
                  f"The biggest lever. The {len(f):,} labeled rows are only about {round(n_arr, -2):,} distinct "
                  f"arrivals over {f.date.nunique()} days, and {swing}'s average delay was {day[swing, lo]:.0f} s on "
                  f"{md(lo)} but {day[swing, hi]:.0f} s on {md(hi)}. More test days also give real error bars, which "
                  "narrow with the square root of the number of days",
                  "Keep the scrapers running daily on all routes at once, through a semester"),
              row("GT Parking & Transportation (RideSystems)", "Possibly months of past GPS and arrivals at once",
                  "The public API has no history, but the dispatch side may"),
              row("TransLoc fields not saved yet",
                  "`IsDelayed`, `OnTimeStatus`, `Heading`, `IsOnRoute`, and occupancy from `GetVehicleCapacities` "
                  "(riders on board out of capacity)",
                  "Same API the scrapers already call; check that occupancy is filled in during the day. Also fix "
                  "Clough's stop-ID key and ETAs of 0 s saved as blank"),
              row("A fixed stop table",
                  "One row per route stop: ID, name, coordinates, order, distance along the route, planned travel and "
                  "dwell times, plus hand-added facts such as a traffic light, crosswalk or major building nearby",
                  "Stops rarely change, so it is built once. The route config in `raw_data/route_config/` already has "
                  "most columns. Route IDs change when a route is redrawn, so version the table by date"),
              row("Google traffic (Routes API)",
                  "Traffic-aware travel time from the bus's current GPS position to the target stop, laid over the route",
                  "Live only, with no way to backfill history, so start collecting it alongside the scrapers. Paid per "
                  "request, with terms on storing results; cache it by road segment"),
              row("A deterministic baseline",
                  "An ETA computed from known quantities: distance left along the route ÷ recent speed, plus planned "
                  "dwell at each stop on the way. The model then only has to learn the leftover error",
                  "Built from the stop table and GPS; no new data needed"),
              row("Campus calendar", "Class-change times, game days, events, breaks",
                  "Explains bad days the current features can't see"),
              row("Live weather", "Current conditions and forecast",
                  "This study used Open-Meteo's historical archive, which isn't available in real time"),
              "", "### 5.3 Which model, and what would change it", "",
              "- **Now:** a gradient-boosted tree model (XGBoost, LightGBM or HistGradientBoosting). Tree ensembles won "
              "on every dataset, and a gradient-boosted model was best or within 3% of the best (section 6). They train "
              "in minutes and handle mixed features well. Ship the lookup table first as the simplest win, and keep it "
              "as the fallback when a feature feed is down. Shelve the NN.",
              "- **If we get much more data** (months, all routes):",
              "  - NNs become worth revisiting, including sequence models that read a bus's recent GPS trace.",
              "  - Per-route models may start to win (5.1).",
              "  - Predicting a range (\"arrives in 6–9 min\") instead of one number becomes realistic, because there "
              "would be enough days to check that the ranges hold.",
              "  - Retrain on a schedule, for example weekly, with day-by-day evaluation.",
              "- **If it has to run live at low latency** (a prediction for every bus and stop on every poll, every "
              "15–30 s):",
              "  - The lookup table is instant, and a gradient-boosted model takes milliseconds per batch; cap its tree "
              "count and depth. Large ExtraTrees forests and 5-seed NN ensembles cost more.",
              "  - Features, not models, are the bottleneck. Rolling features (2-minute speed, time since the last "
              "arrival) need live per-bus state, weather needs a live feed, and Google calls add network delay and "
              "cost, so cache them and refresh every few minutes.",
              "  - Plan for missing inputs: fall back to the lookup table, then to TransLoc's ETA.",
              "- **Open question:** whether accuracy (more data) or a live product (latency) comes first decides "
              "whether the next effort goes into modeling or engineering."]

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
