"""Build the bus / route / combined datasets from the raw March 2026 scrapes.

Target  delay_s = actual arrival - TransLoc ETA (seconds; positive = bus later than TransLoc said).
        The actual arrival comes from GPS: the moment a bus passes within 25 m of the stop.
Split   Whole service days are held out: Mar 4 (the first day with Gold, Red and Clough all recorded)
        and Mar 9 (the first Green day). Every route keeps later days for training; test share 26-32%.
        Chosen for coverage before any model was scored on these days.
Output  datasets/<ds>_features.parquet + <ds>_features_schema.json + build_report.md

Run:    python build_datasets.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit

MC = Path(__file__).resolve().parents[1]
RAW, OUT = MC / "raw_data", MC / "datasets"
R_M = 25              # metres: bus counts as "at the stop"
MAX_GAP_S = 300       # polling gap that breaks a trajectory
OFF_ROUTE_M = 50      # farther than this from the route path = not in service
MAX_ACTUAL_S = 2700   # ignore matches more than 45 min ahead
SEED = 42
ROUTES = {29: "Gold", 17: "Green", 20: "Red", 28: "Clough"}
TEST_DAYS = {"2026-03-04", "2026-03-09"}
META = ["t_local", "date", "route_name", "route_id", "vehicle_id", "route_stop_id", "split", "cv_group"]
TARGET = "delay_s"

report = []


def log(msg=""):
    print(msg)
    report.append(msg)


def xy(lat, lon):
    """Local planar metres around GT (equirectangular; fine at campus scale)."""
    return np.c_[(np.asarray(lon, float) + 84.39) * 111320 * np.cos(np.radians(33.777)),
                 (np.asarray(lat, float) - 33.777) * 110540]


# ------------------------------------------------------------------ route config
CFG = {r["RouteID"]: r for r in json.loads(
    (RAW / "route_config" / "GetRoutesForMapWithSchedule_2026-09-10.json").read_text(encoding="utf-8"))}


def route_meta(rid):
    stops = sorted(CFG[rid]["Stops"], key=lambda s: s["Order"])   # Order has gaps (1, 5, 10, ...): rank it
    st = pd.DataFrame({"route_stop_id": [s["RouteStopID"] for s in stops],
                       "stop_lat": [s["Latitude"] for s in stops], "stop_lon": [s["Longitude"] for s in stops],
                       "rank": range(len(stops)), "dwell_s": [s["SecondsAtStop"] for s in stops],
                       "to_next_s": [s["SecondsToNextStop"] for s in stops]})
    path = xy([p["Latitude"] for s in stops for p in s["MapPoints"]], [p["Longitude"] for s in stops for p in s["MapPoints"]])
    return st, path


# ------------------------------------------------------------------ T2.1 load & normalise
def load_raw(rid):
    if rid in (29, 17):
        folder = RAW / ("gold/eta_scraper" if rid == 29 else "green")
        df = pd.concat([pd.read_csv(f) for f in sorted(folder.glob("*.csv"))], ignore_index=True)
        t = pd.to_datetime(df.snapshot_time, format="ISO8601")
        eta_s = (pd.to_datetime(df.estimated_time_arrival, format="ISO8601") - t).dt.total_seconds()
    elif rid == 20:
        df = pd.read_csv(RAW / "red/red_line_data.csv")
        t = pd.to_datetime(df.snapshot_time_utc, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        eta_s = df.eta_seconds.astype(float)
    else:
        df = pd.read_csv(RAW / "clough/clough_bus_data.csv")
        t = pd.to_datetime(df.snapshot_time.str.replace(" EST", "", regex=False))
        ms = df.estimated_time_arrival.str.extract(r"(\d+)")[0].astype(float)   # /Date(ms)/ is true UTC
        eta_local = pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
        eta_s = (eta_local - t).dt.total_seconds().clip(lower=0)               # -2 s = request latency
    out = pd.DataFrame({"t": t.astype("datetime64[ns]").values, "vehicle_id": df.vehicle_id.values, "route_id": rid,
                        "route_stop_id": pd.to_numeric(df.route_stop_id, errors="coerce").values,
                        "lat": df.vehicle_lat.values, "lon": df.vehicle_lon.values,
                        "speed": df.vehicle_speed.values.astype(float), "eta_s": eta_s.values})
    return out


# ------------------------------------------------------------------ T2.3 arrival detection
def detect_arrivals(pos, st):
    """One vehicle's time-sorted positions -> DataFrame(route_stop_id, arrival, known_at) in route order.

    arrival  = interpolated pass-by time (used only for labels).
    known_at = when the arrival becomes observable: the GPS fix that closes the detecting segment, or, for
               an out-of-order visit the route-order filter accepts only once the next visit confirms it,
               that confirming fix. Features may only use arrivals with known_at <= t.
    Visits are processed in the order they become observable (ties broken by time, then stop id), so the
    filter's decisions never depend on data from after that moment.
    """
    rank = dict(zip(st.route_stop_id, st["rank"]))
    n, ids = len(st), st.route_stop_id.to_numpy()
    S, P = xy(st.stop_lat, st.stop_lon), xy(pos.lat, pos.lon)
    t_ns = pos.t.to_numpy().astype("datetime64[ns]").astype(np.int64)     # exact integer ns (no float rounding)
    ok = np.diff(t_ns) <= MAX_GAP_S * 10**9
    P0, P1, t0, t1 = P[:-1][ok], P[1:][ok], t_ns[:-1][ok], t_ns[1:][ok]
    d = P1 - P0
    w = S[None] - P0[:, None]
    f = np.clip((w * d[:, None]).sum(-1) / ((d ** 2).sum(-1)[:, None] + 1e-9), 0, 1)
    dist = np.linalg.norm(w - f[..., None] * d[:, None], axis=-1)
    hp, hs = np.nonzero(dist < R_M)
    if not len(hp):
        return pd.DataFrame(columns=["route_stop_id", "arrival", "known_at"])
    tt = t0[hp] + np.round(f[hp, hs] * (t1 - t0)[hp]).astype(np.int64)    # within [t0, t1] by construction
    det = pd.DataFrame({"route_stop_id": ids[hs], "tt": tt, "seen": t1[hp]})
    det = det.sort_values(["route_stop_id", "tt"], kind="stable")
    det["visit"] = (det.groupby("route_stop_id").tt.diff().fillna(10**18) > 120 * 10**9).cumsum()
    visits = det.groupby("visit").agg(route_stop_id=("route_stop_id", "first"), tt=("tt", "min"), seen=("seen", "min"))
    visits = visits.sort_values(["seen", "tt", "route_stop_id"], kind="stable")
    kept, last, pending = [], None, None     # keep visits that advance 1-4 stops in route order
    for v in visits.itertuples():
        step = lambda a: (rank[v.route_stop_id] - rank[a.route_stop_id]) % n
        if last is None or 1 <= step(last) <= 4:
            kept.append((v.route_stop_id, v.tt, v.seen)); last, pending = v, None
        elif pending is not None and 1 <= step(pending) <= 4:
            kept += [(pending.route_stop_id, pending.tt, v.seen), (v.route_stop_id, v.tt, v.seen)]; last, pending = v, None
        else:
            pending = v
    out = pd.DataFrame(kept, columns=["route_stop_id", "tt", "known_ns"])
    out["arrival"] = pd.to_datetime(out.tt.astype(np.int64)).astype("datetime64[ns]")
    out["known_at"] = pd.to_datetime(out.known_ns.astype(np.int64)).astype("datetime64[ns]")
    return out[["route_stop_id", "arrival", "known_at"]].reset_index(drop=True)


def segment_of(pos_t, times, seg):
    idx = np.clip(np.searchsorted(pos_t, times, side="right") - 1, 0, len(seg) - 1)
    return seg[idx]


# ------------------------------------------------------------------ T2.4 bus-state features (past-only)
def bus_state(pos):
    """Per vehicle, per poll: rolling speed, time stationary, stale-GPS flag."""
    pos = pos.sort_values("t").copy()
    moved = (pos.lat.diff() != 0) | (pos.lon.diff() != 0)
    pos["stationary_s"] = (pos.t - pos.t.where(moved).ffill().fillna(pos.t.iloc[0])).dt.total_seconds()
    gap = pos.t.diff().dt.total_seconds()
    pos["stale_gps"] = ((~moved) & (pos.speed > 0) & (gap <= 90)).astype(int)
    pos["speed_mean_2min"] = pos.set_index("t").speed.rolling("120s").mean().to_numpy()
    return pos[["t", "vehicle_id", "stationary_s", "stale_gps", "speed_mean_2min"]]


def eta_sequence_features(rows, key):
    """ETA change vs the previous poll (<= 90 s earlier) and how many polls it has been frozen."""
    rows = rows.sort_values(key + ["t"]).copy()
    g = rows.groupby(key, sort=False)
    prev_eta, prev_t = g.eta_s.shift(), g.t.shift()
    has_prev = prev_eta.notna() & ((rows.t - prev_t).dt.total_seconds() <= 90)
    rows["eta_delta_s"] = np.where(has_prev, rows.eta_s - prev_eta, 0.0)
    rows["eta_prev_missing"] = (~has_prev).astype(int)
    frozen = (has_prev & (rows.eta_delta_s == 0)).astype(int)
    run = (frozen == 0).groupby([rows[k] for k in key]).cumsum()
    rows["eta_frozen_polls"] = frozen.groupby([rows[k] for k in key] + [run]).cumsum()
    return rows


# ------------------------------------------------------------------ per-route pipeline
def build_route(rid, weather):
    name = ROUTES[rid]
    st, path = route_meta(rid)
    stop_xy = dict(zip(st.route_stop_id, map(tuple, xy(st.stop_lat, st.stop_lon))))
    raw = load_raw(rid)
    log(f"\n### {name} (route {rid})")
    log(f"- raw rows: {len(raw):,}")

    # positions from ALL rows (dwell points matter for arrival detection)
    pos_all = raw.drop_duplicates(["t", "vehicle_id"]).sort_values(["vehicle_id", "t"])
    n_active = pos_all.groupby("t").vehicle_id.nunique().rename("n_buses_active")
    arrivals = {v: detect_arrivals(p, st) for v, p in pos_all.groupby("vehicle_id")}
    log(f"- detected stop arrivals: {sum(len(a) for a in arrivals.values()):,}")
    state = pd.concat([bus_state(p) for _, p in pos_all.groupby("vehicle_id")])

    # T2.2 cleaning
    rows = raw.drop_duplicates()
    log(f"- exact duplicate rows dropped: {len(raw) - len(rows):,}")
    next_stop_only = rid in (20, 28)
    key = ["vehicle_id"] if rid == 28 else ["vehicle_id", "route_stop_id"]
    rows = eta_sequence_features(rows, key)
    n0 = len(rows)
    rows = rows[~(rows.route_stop_id.isna() & rows.eta_s.isna())]
    log(f"- placeholder rows (no stop, no ETA) dropped: {n0 - len(rows):,}")
    n0 = len(rows)
    rows = rows[rows.eta_s.notna()]
    why = "bus off its route" if rid == 20 else "bus at the stop (0-s ETA saved as blank)"
    log(f"- blank-ETA rows dropped ({why}): {n0 - len(rows):,}")
    rows["off_route_m"] = cKDTree(path).query(xy(rows.lat, rows.lon))[0]
    n0 = len(rows)
    rows = rows[rows.off_route_m <= OFF_ROUTE_M]
    log(f"- off-route rows (> {OFF_ROUTE_M} m from route) dropped: {n0 - len(rows):,}")
    n0 = len(rows)
    rows = rows[rows.t.dt.hour >= 6]
    log(f"- overnight rows (00:00-05:59) dropped: {n0 - len(rows):,}")
    if rid != 28:
        rows["route_stop_id"] = rows.route_stop_id.astype(int)

    # T2.4 causal "last known arrival" for every row: the latest stop visit observable at t
    rank = dict(zip(st.route_stop_id, st["rank"]))
    n = len(st)
    segs, parts = {}, []
    for vid, g in rows.groupby("vehicle_id"):
        arr = arrivals.get(vid)
        if arr is None or arr.empty:
            continue
        pos = pos_all[pos_all.vehicle_id == vid]
        pt = pos.t.to_numpy()
        seg = (pos.t.diff().dt.total_seconds() > MAX_GAP_S).cumsum().to_numpy()
        segs[vid] = (pt, seg)
        g = g.sort_values("t").copy()
        g["seg"] = segment_of(pt, g.t.to_numpy(), seg)
        known = arr.sort_values(["known_at", "arrival"]).rename(columns={"route_stop_id": "last_stop", "arrival": "last_arrival"})
        known["seg_l"] = segment_of(pt, known.last_arrival.to_numpy(), seg)
        m = pd.merge_asof(g, known, left_on="t", right_on="known_at", direction="backward")
        parts.append(m[m.seg == m.seg_l])
    n0 = len(rows)
    rows = pd.concat(parts, ignore_index=True)
    assert (rows.known_at <= rows.t).all() and (rows.last_arrival <= rows.t).all(), "feature uses a future arrival"
    rows["last_stop"] = rows.last_stop.astype(int)
    log(f"- rows without a known prior stop arrival in the same trajectory dropped: {n0 - len(rows):,}")

    # Target stop. Next-stop scrapers report TransLoc's next stop. Clough lost its stop ID, so use the causal
    # guess "the stop after the last known arrival", checked on Red where TransLoc's stop ID is present.
    if next_stop_only:
        order = st.route_stop_id.to_numpy()
        guess = rows.last_stop.map({s: int(order[(r + 1) % n]) for s, r in rank.items()})
        # while dwelling at a stop, "next stop" can mean the current one (ETA 0) or the following one
        dwelling = np.linalg.norm(xy(rows.lat, rows.lon)[:, None] - xy(st.stop_lat, st.stop_lon)[None], axis=-1).min(1) < R_M
        if rid == 20:   # Red keeps TransLoc's real stop IDs; it only serves to check the guess
            a = float((guess[~dwelling] == rows.route_stop_id[~dwelling]).mean())
            log(f"- **causal next-stop check on Red**: 'stop after the last known arrival' = TransLoc's stop for "
                f"{a:.1%} of {int((~dwelling).sum()):,} rows with the bus between stops")
            ROUTE_CHECK["red_inference_agreement"] = a
        else:
            rows = rows[~dwelling].assign(route_stop_id=guess[~dwelling].astype(int))
            log(f"- rows with the bus dwelling at a stop (ambiguous next stop) dropped: {int(dwelling.sum()):,}")
    # Stale ETA: TransLoc still points at the stop the bus just passed (next-stop feed, or ETA < 2 min).
    # Its ETA refers to that visit, but the only later arrival is a lap away -> a fake ~30 min delay.
    stale = (rows.route_stop_id == rows.last_stop) & (next_stop_only | (rows.eta_s < 120))
    rows = rows[~stale]
    log(f"- stale-ETA rows (target = the stop just passed) dropped: {int(stale.sum()):,}")
    tgt = st.set_index("route_stop_id").loc[rows.route_stop_id]
    at_stop = np.linalg.norm(xy(rows.lat, rows.lon) - xy(tgt.stop_lat, tgt.stop_lon), axis=1) < R_M
    rows = rows[~at_stop]
    log(f"- rows with the bus already at the target stop (ambiguous ETA) dropped: {int(at_stop.sum()):,}")

    # T2.3 labels: this bus's next arrival at the target stop after t (labels may use the future)
    labeled = []
    for vid, g in rows.groupby("vehicle_id"):
        pt, seg = segs[vid]
        arr = arrivals[vid].sort_values("arrival").drop(columns="known_at")
        arr["seg_a"] = segment_of(pt, arr.arrival.to_numpy(), seg)
        m = pd.merge_asof(g.sort_values("t"), arr, left_on="t", right_on="arrival", by="route_stop_id",
                          direction="forward", allow_exact_matches=False)
        labeled.append(m[m.seg == m.seg_a])
    lab = pd.concat(labeled, ignore_index=True)
    log(f"- rows with no later arrival at the target stop in the same trajectory dropped: {len(rows) - len(lab):,}")
    lab["actual_s"] = (lab.arrival - lab.t).dt.total_seconds()
    n0 = len(lab)
    lab = lab[(lab.actual_s > 0) & (lab.actual_s <= MAX_ACTUAL_S)]
    log(f"- labeled rows: {len(lab):,} (dropped {n0 - len(lab):,} with actual > {MAX_ACTUAL_S}s)")
    lab["stops_ahead"] = [((rank[s] - rank[l]) % n) or n for s, l in zip(lab.route_stop_id, lab.last_stop)]
    per = (st.dwell_s + st.to_next_s).to_numpy()
    cum = np.concatenate([[0], np.cumsum(np.concatenate([per, per]))])
    k = np.array([rank[l] for l in lab.last_stop])
    lab["planned_s_ahead"] = cum[k + lab.stops_ahead.to_numpy()] - cum[k]
    lab["since_last_arrival_s"] = (lab.t - lab.last_arrival).dt.total_seconds()
    lab["planned_remaining_s"] = (lab.planned_s_ahead - lab.since_last_arrival_s).clip(lower=0)

    # stop static + geometry
    lab = lab.merge(st[["route_stop_id", "stop_lat", "stop_lon", "rank", "dwell_s"]], on="route_stop_id", how="left")
    lab["stop_rank_frac"] = lab["rank"] / n
    lab["stop_planned_dwell_s"] = lab.dwell_s
    lab["dist_to_stop_m"] = np.linalg.norm(xy(lab.lat, lab.lon) - xy(lab.stop_lat, lab.stop_lon), axis=1)

    # bus state, fleet size, time, weather
    lab = lab.merge(state, on=["t", "vehicle_id"], how="left").merge(n_active, left_on="t", right_index=True, how="left")
    lab["hour"] = lab.t.dt.hour
    lab["minute_of_day"] = lab.t.dt.hour * 60 + lab.t.dt.minute
    ang = 2 * np.pi * (lab.minute_of_day + lab.t.dt.second / 60) / 1440
    lab["tod_sin"], lab["tod_cos"] = np.sin(ang), np.cos(ang)
    lab["dow"] = lab.t.dt.day_name()
    lab["wx_hour"] = lab.t.dt.floor("h")
    lab = lab.merge(weather, left_on="wx_hour", right_index=True, how="left")
    assert lab[weather.columns].notna().all().all(), "weather join left gaps"

    lab[TARGET] = lab.actual_s - lab.eta_s
    lab["route_name"] = name
    lab = lab.rename(columns={"t": "t_local", "lat": "vehicle_lat", "lon": "vehicle_lon", "speed": "vehicle_speed"})
    lab["date"] = lab.t_local.dt.date.astype(str)
    log(f"- final rows: {len(lab):,}; days: {sorted(lab.date.unique())}; vehicles: {lab.vehicle_id.nunique()}")
    log(f"- delay_s: median {lab[TARGET].median():+.0f}s, mean {lab[TARGET].mean():+.0f}s, "
        f"p5 {lab[TARGET].quantile(.05):+.0f}s, p95 {lab[TARGET].quantile(.95):+.0f}s")
    return lab


ROUTE_CHECK = {}


# ------------------------------------------------------------------ T2.5 weather
def wx_group(code):
    if code in (0, 1): return "clear"
    if code in (2, 3): return "cloudy"
    if code in (45, 48): return "fog"
    if 51 <= code <= 57: return "drizzle"
    if 61 <= code <= 67 or 80 <= code <= 82: return "rain"
    if 71 <= code <= 77 or code in (85, 86): return "snow"
    if code >= 95: return "thunder"
    return "other"


def load_weather():
    # Fetched in UTC: the archive's local-time option applies one fixed offset to the whole range, which
    # would shift everything before the Mar 8 DST change by an hour. Convert with the real tz rules instead.
    w = json.loads((RAW / "weather" / "open_meteo_hourly_utc_2026-03-02_2026-03-17.json").read_text(encoding="utf-8"))["hourly"]
    df = pd.DataFrame(w)
    utc = pd.to_datetime(df.pop("time")).dt.tz_localize("UTC")
    df.index = pd.DatetimeIndex(utc.dt.tz_convert("America/New_York").dt.tz_localize(None)).astype("datetime64[ns]")
    assert df.index.is_unique
    df = df.rename(columns={"temperature_2m": "temp_f", "apparent_temperature": "apparent_temp_f",
                            "relative_humidity_2m": "rel_humidity", "precipitation": "precip_in", "rain": "rain_in",
                            "cloud_cover": "cloud_cover", "wind_speed_10m": "wind_mph", "wind_gusts_10m": "wind_gust_mph"})
    df["precip_prev_1h"] = df.precip_in.shift(1).fillna(0)
    df["is_raining"] = (df.precip_in > 0).astype(int)
    grp = df.pop("weather_code").astype(int).map(wx_group)
    df = df.join(pd.get_dummies(grp, prefix="wx").astype(int))
    return df


# ------------------------------------------------------------------ T2.6 embeddings
def fit_embeddings(df, feats, id_cols, seed=SEED):
    """Entity embeddings for ID columns, learned on TRAIN rows only (small MLP predicting delay)."""
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(8)
    tr = df[df.split == "train"]
    itr, iva = next(GroupShuffleSplit(1, test_size=0.1, random_state=seed).split(tr, groups=tr.cv_group))
    X = tr[feats].to_numpy(np.float32)
    mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-6
    y = tr[TARGET].to_numpy(np.float32)
    ym, ys = y[itr].mean(), y[itr].std() + 1e-6
    maps = {c: {v: i for i, v in enumerate(sorted(tr[c].unique()))} for c in id_cols}
    dims = {c: 4 if c == "route_stop_id" else min(4, max(2, len(maps[c]) // 2)) for c in id_cols}
    emb = nn.ModuleDict({c: nn.Embedding(len(maps[c]), dims[c]) for c in id_cols})
    mlp = nn.Sequential(nn.Linear(len(feats) + sum(dims.values()), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    params = list(emb.parameters()) + list(mlp.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    xt = torch.tensor((X - mu) / sd)
    ids = torch.tensor(np.stack([tr[c].map(maps[c]).to_numpy() for c in id_cols], 1), dtype=torch.long)
    yt = torch.tensor((y - ym) / ys)
    fwd = lambda i: mlp(torch.cat([xt[i]] + [emb[c](ids[i, k]) for k, c in enumerate(id_cols)], 1)).squeeze(1)
    best, best_state, bad = np.inf, None, 0
    for epoch in range(60):
        perm = torch.tensor(np.random.permutation(itr))
        for b in range(0, len(perm), 512):
            i = perm[b:b + 512]
            opt.zero_grad(); nn.functional.mse_loss(fwd(i), yt[i]).backward(); opt.step()
        with torch.no_grad():
            v = nn.functional.mse_loss(fwd(torch.tensor(iva)), yt[iva]).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {c: emb[c].weight.detach().clone() for c in id_cols}
        elif (bad := bad + 1) >= 6:
            break
    cols = {}
    for c in id_cols:
        W = best_state[c].numpy()
        idx = df[c].map(maps[c])
        vec = np.where(idx.notna().to_numpy()[:, None], W[idx.fillna(0).astype(int).to_numpy()], W.mean(0))
        for j in range(dims[c]):
            cols[f"{'vehicle' if c == 'vehicle_id' else 'stop'}_emb_{j}"] = vec[:, j]
    log(f"  embeddings {dims} trained {epoch + 1} epochs (val MSE std-units {best:.3f})")
    return pd.DataFrame(cols, index=df.index)


# ------------------------------------------------------------------ T2.7 assemble
BASE_FEATURES = ["eta_s", "eta_delta_s", "eta_prev_missing", "eta_frozen_polls", "dist_to_stop_m", "stops_ahead",
                 "planned_s_ahead", "since_last_arrival_s", "planned_remaining_s", "vehicle_speed", "speed_mean_2min",
                 "stationary_s", "stale_gps", "vehicle_lat", "vehicle_lon", "n_buses_active", "hour", "minute_of_day",
                 "tod_sin", "tod_cos", "stop_lat", "stop_lon", "stop_rank_frac", "stop_planned_dwell_s"]


def make_dataset(ds, df, weather_cols):
    """Whole service days are held out (labels reach 45 min ahead and delays drift within a day, so
    interleaved hour blocks leak). TEST_DAYS keep every route in both splits (see the module docstring)."""
    df = df.copy().reset_index(drop=True)
    df["split"] = np.where(df.date.isin(TEST_DAYS), "test", "train")
    train_days = df.loc[df.split == "train", "date"].nunique()
    # Tuning groups inside train mirror the test: whole days whenever there are >= 2 train days (within-day
    # blocks reward day-specific quirks - on `route` they picked linear models that failed on the new day).
    # With a single train day (`bus`) only 3-hour blocks are possible (a known, tuning-only compromise).
    df["cv_group"] = df.date if train_days >= 2 else df.date + "_" + (df.t_local.dt.hour // 3 * 3).astype(str).str.zfill(2)
    onehots = pd.get_dummies(df.dow, prefix="dow").astype(int)
    if ds == "combined":
        onehots = onehots.join(pd.get_dummies(df.route_name, prefix="route").astype(int))
    df = df.join(onehots)
    train = df.split == "train"
    feats = BASE_FEATURES + weather_cols + list(onehots.columns)
    feats = [c for c in feats if df.loc[train, c].nunique() > 1]          # constant in train -> useless
    id_cols = [c for c in ("vehicle_id", "route_stop_id") if df.loc[train, c].nunique() > 1]
    log(f"\n### dataset `{ds}`")
    df = df.join(fit_embeddings(df, feats, id_cols))
    feats += [c for c in df.columns if "_emb_" in c]
    feats = [c for c in feats if df.loc[train, c].nunique() > 1]
    assert set(df.loc[train, "route_name"]) == set(df.loc[~train, "route_name"]), "every route needs train and test days"
    X = df[feats + [TARGET]]
    assert np.isfinite(X.to_numpy(float)).all(), "NaN/inf in features or target"
    assert not set(feats) & set(META + ["actual_s", "arrival", "last_arrival"]), "meta/label leaked into features"
    df[META + feats + [TARGET]].to_parquet(OUT / f"{ds}_features.parquet", index=False)
    schema = {"dataset": ds, "target": TARGET, "features": feats, "meta": META, "split_col": "split",
              "group_col": "cv_group", "n_rows": len(df), "created": datetime.now().isoformat(timespec="seconds"),
              "target_definition": "delay_s = actual arrival (GPS pass-by within 25 m of stop) - TransLoc ETA, seconds; positive = later than TransLoc said"}
    (OUT / f"{ds}_features_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    sp = df.split.value_counts()
    log(f"- rows {len(df):,} (train {sp.get('train', 0):,} / test {sp.get('test', 0):,}); "
        f"cv groups {df.cv_group.nunique()} (test {df[df.split == 'test'].cv_group.nunique()}); features {len(feats)}")
    log(f"- train days {sorted(df[train].date.unique())}; test days {sorted(df[~train].date.unique())}; "
        f"vehicles: {df.vehicle_id.nunique()}; stops: {df.route_stop_id.nunique()}")
    assert min(sp.get("train", 0), sp.get("test", 0)) >= 0.1 * len(df)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("# Build report\n")
    log(f"Built {datetime.now():%Y-%m-%d %H:%M}. Target: `delay_s` = actual arrival - TransLoc ETA (s).")
    weather = load_weather()
    per_route = {rid: build_route(rid, weather) for rid in ROUTES}
    agree = ROUTE_CHECK.get("red_inference_agreement", 0)
    if agree < 0.8:
        log(f"\n**Clough excluded from combined: causal next-stop agreement on Red only {agree:.1%} (< 80%).**")
        per_route.pop(28)
    wcols = list(weather.columns)
    make_dataset("combined", pd.concat(per_route.values(), ignore_index=True), wcols)
    green = per_route[17]
    make_dataset("route", green, wcols)
    # the bus must have both a training day and a held-out day; take the one with the most rows
    split_days = green.assign(test=green.date.isin(TEST_DAYS)).groupby("vehicle_id").test.agg(["min", "max"])
    both = split_days[(~split_days["min"]) & split_days["max"]].index
    counts = green[green.vehicle_id.isin(both)].vehicle_id.value_counts()
    bus = int(counts.index[0])
    log(f"\nChosen bus for `bus` dataset: Green vehicle {bus} ({counts.iloc[0]:,} rows; "
        f"days {sorted(green[green.vehicle_id == bus].date.unique())})")
    make_dataset("bus", green[green.vehicle_id == bus], wcols)
    (OUT / "build_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
