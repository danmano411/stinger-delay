# Results

Test days are held out whole (Mar 4 and Mar 9), and each model is scored on them once. Error is the mean absolute error (MAE) of the predicted arrival time, in seconds. Predicting `delay_s` is the same as correcting TransLoc's ETA, so "TransLoc as-is" is simply a prediction of zero delay.

**Best model** means the tuned model with the lowest test MAE on that dataset: HistGradientBoostingRegressor for `bus`, ExtraTreesRegressor for `route`, XGBRegressor for `combined`. It is picked on the test days, so its score is slightly optimistic. Percentages are computed from the rounded numbers shown.

## 1. Best model vs TransLoc, and where the gain comes from

**L, the lookup table,** is a correction built for this study, not something TransLoc provides. It looks only at TransLoc's ETA. The training rows are sorted by ETA and cut into 10 ranges with the same number of rows each, and each range stores the average delay seen in training. To predict, find the range the ETA falls in and add that range's average delay. For example, on `route` one range is 6.1–8.1 min with an average delay of +7.9 min, so when TransLoc says 7 min, L predicts 14.9 min. The best model (**M**) adds the rest of the gain using the other features.

|  | `bus` | `route` | `combined` |
| --- | --- | --- | --- |
| Best model | HistGradientBoostingRegressor | ExtraTreesRegressor | XGBRegressor |
| **T** TransLoc as-is | 279 s | 290 s | 256 s |
| **L** Lookup table on TransLoc's ETA | 183 s | 173 s | 178 s |
| **M** Best model | 127 s | 136 s | 143 s |
| **Error reduction** (T − M) / T | **(279 − 127) / 279 = 54%** | **(290 − 136) / 290 = 53%** | **(256 − 143) / 256 = 44%** |
| … from the lookup table (T − L) / T | (279 − 183) / 279 = 34% | (290 − 173) / 290 = 40% | (256 − 178) / 256 = 30% |
| … added by the model (L − M) / T | (183 − 127) / 279 = 20% | (173 − 136) / 290 = 13% | (178 − 143) / 256 = 14% |

Two more baselines (the training mean delay, and a straight line in the ETA) are in section 6.

## 2. Error by how far away TransLoc says the bus is

| Dataset (best model) | TransLoc's ETA | Rows | TransLoc error | Model error | Reduction (TransLoc − model) / TransLoc |
| --- | --- | --- | --- | --- | --- |
| `bus` (HistGradientBoostingRegressor) | under 2 min | 1,377 | 99 s | 83 s | (99 − 83) / 99 = 16% |
|  | 2–5 min | 1,722 | 208 s | 103 s | (208 − 103) / 208 = 50% |
|  | 5–10 min | 2,350 | 308 s | 134 s | (308 − 134) / 308 = 56% |
|  | 10–20 min | 2,087 | 423 s | 169 s | (423 − 169) / 423 = 60% |
|  | **all** | 7,536 | 279 s | 127 s | (279 − 127) / 279 = 54% |
| `route` (ExtraTreesRegressor) | under 2 min | 3,085 | 103 s | 66 s | (103 − 66) / 103 = 36% |
|  | 2–5 min | 3,927 | 203 s | 106 s | (203 − 106) / 203 = 48% |
|  | 5–10 min | 5,597 | 340 s | 144 s | (340 − 144) / 340 = 58% |
|  | 10–20 min | 4,389 | 438 s | 201 s | (438 − 201) / 438 = 54% |
|  | **all** | 16,998 | 290 s | 136 s | (290 − 136) / 290 = 53% |
| `combined` (XGBRegressor) | under 2 min | 8,646 | 88 s | 74 s | (88 − 74) / 88 = 16% |
|  | 2–5 min | 5,184 | 178 s | 92 s | (178 − 92) / 178 = 48% |
|  | 5–10 min | 6,901 | 316 s | 154 s | (316 − 154) / 316 = 51% |
|  | 10–20 min | 6,868 | 418 s | 214 s | (418 − 214) / 418 = 49% |
|  | 20+ min | 1,542 | 463 s | 328 s | (463 − 328) / 463 = 29% |
|  | **all** | 29,141 | 256 s | 143 s | (256 − 143) / 256 = 44% |

**The models help most from 2 to 20 minutes out**, which is likely when riders decide whether to head to the stop (an assumption; there is no rider data here). In that range the error falls by 48–60% and the share of predictions within 2 minutes of the actual arrival rises by 23–34 points (section 3). Under 2 minutes TransLoc is already fairly close (16–36% cut), and beyond 20 minutes (`combined` only) the cut is 29%.

TransLoc's error is almost all lateness. On average the bus arrives 1.4 to 1.7 min later than TransLoc says when it is under 2 min away, and 6.7 to 7.3 min later when it is 10–20 min away. That steady, growing optimism is what the lookup table in section 1 corrects.

## 3. How often the prediction is within 2 minutes of the actual arrival

| Dataset (best model) | TransLoc's ETA | TransLoc | Model | Gain |
| --- | --- | --- | --- | --- |
| `bus` (HistGradientBoostingRegressor) | under 2 min | 70% | 79% | 79 − 70 = +9 pts |
|  | 2–5 min | 39% | 66% | 66 − 39 = +27 pts |
|  | 5–10 min | 23% | 52% | 52 − 23 = +29 pts |
|  | 10–20 min | 9% | 43% | 43 − 9 = +34 pts |
|  | **all** | 31% | 58% | 58 − 31 = +27 pts |
| `route` (ExtraTreesRegressor) | under 2 min | 71% | 81% | 81 − 71 = +10 pts |
|  | 2–5 min | 39% | 64% | 64 − 39 = +25 pts |
|  | 5–10 min | 17% | 51% | 51 − 17 = +34 pts |
|  | 10–20 min | 8% | 31% | 31 − 8 = +23 pts |
|  | **all** | 29% | 55% | 55 − 29 = +26 pts |
| `combined` (XGBRegressor) | under 2 min | 82% | 88% | 88 − 82 = +6 pts |
|  | 2–5 min | 46% | 75% | 75 − 46 = +29 pts |
|  | 5–10 min | 20% | 48% | 48 − 20 = +28 pts |
|  | 10–20 min | 13% | 44% | 44 − 13 = +31 pts |
|  | 20+ min | 11% | 26% | 26 − 11 = +15 pts |
|  | **all** | 41% | 63% | 63 − 41 = +22 pts |

## 4. By route, and pooled vs single-route models

- **The gain is uneven by route.** On the `combined` test days, the best model's error by route is Gold 298 → 228 s (23%), Green 290 → 126 s (57%), Red 84 → 80 s (5%). Red's feed only gives next-stop ETAs, so TransLoc is already close there.
- **Pooling routes helped, even for a single bus.** On identical test rows, the `combined` model beat the models trained on one route or one bus: on Green's test day 126 s vs 136 s for the best `route` model, and on bus #3's rows 114 s vs 127 s for the best `bus` model. `combined` has five training days instead of one or two. This rests on one test day, so treat it as a first sign.

## 5. Next steps

### 5.1 Model structure: `route` or `combined`, decided by the data

- **Drop per-bus models.** A live system needs a prediction for every bus, one bus has the least data, and on bus #3's own rows the pooled model was better (section 4). Bus identity works better as a feature inside a larger model, as the bus embeddings already are.
- **Choose between `route` and `combined` from the data we collect.** Today `combined` wins on Green (section 4), because pooling adds days no single route has. If each route gets many days of its own, per-route models may win where routes behave differently (Red's feed, for instance, only gives next-stop ETAs). Decide per route: train both on the same days, evaluate day by day (train on everything before a day, test on that day, repeat), and keep whichever wins. A middle path is one pooled model with a per-route correction on top.

### 5.2 More information to collect

| Source | What it adds | Notes |
| --- | --- | --- |
| More days, all routes on the same days | The biggest lever. The 110,569 labeled rows are only about 4,800 distinct arrivals over 7 days, and Gold's average delay was 18 s on Mar 6 but 356 s on Mar 5. More test days also give real error bars, which narrow with the square root of the number of days | Keep the scrapers running daily on all routes at once, through a semester |
| GT Parking & Transportation (RideSystems) | Possibly months of past GPS and arrivals at once | The public API has no history, but the dispatch side may |
| TransLoc fields not saved yet | `IsDelayed`, `OnTimeStatus`, `Heading`, `IsOnRoute`, and occupancy from `GetVehicleCapacities` (riders on board out of capacity) | Same API the scrapers already call; check that occupancy is filled in during the day. Also fix Clough's stop-ID key and ETAs of 0 s saved as blank |
| A fixed stop table | One row per route stop: ID, name, coordinates, order, distance along the route, planned travel and dwell times, plus hand-added facts such as a traffic light, crosswalk or major building nearby | Stops rarely change, so it is built once. The route config in `raw_data/route_config/` already has most columns. Route IDs change when a route is redrawn, so version the table by date |
| Google traffic (Routes API) | Traffic-aware travel time from the bus's current GPS position to the target stop, laid over the route | Live only, with no way to backfill history, so start collecting it alongside the scrapers. Paid per request, with terms on storing results; cache it by road segment |
| A deterministic baseline | An ETA computed from known quantities: distance left along the route ÷ recent speed, plus planned dwell at each stop on the way. The model then only has to learn the leftover error | Built from the stop table and GPS; no new data needed |
| Campus calendar | Class-change times, game days, events, breaks | Explains bad days the current features can't see |
| Live weather | Current conditions and forecast | This study used Open-Meteo's historical archive, which isn't available in real time |

### 5.3 Which model, and what would change it

- **Now:** a gradient-boosted tree model (XGBoost, LightGBM or HistGradientBoosting). Tree ensembles won on every dataset, and a gradient-boosted model was best or within 3% of the best (section 6). They train in minutes and handle mixed features well. Ship the lookup table first as the simplest win, and keep it as the fallback when a feature feed is down. Shelve the NN.
- **If we get much more data** (months, all routes):
  - NNs become worth revisiting, including sequence models that read a bus's recent GPS trace.
  - Per-route models may start to win (5.1).
  - Predicting a range ("arrives in 6–9 min") instead of one number becomes realistic, because there would be enough days to check that the ranges hold.
  - Retrain on a schedule, for example weekly, with day-by-day evaluation.
- **If it has to run live at low latency** (a prediction for every bus and stop on every poll, every 15–30 s):
  - The lookup table is instant, and a gradient-boosted model takes milliseconds per batch; cap its tree count and depth. Large ExtraTrees forests and 5-seed NN ensembles cost more.
  - Features, not models, are the bottleneck. Rolling features (2-minute speed, time since the last arrival) need live per-bus state, weather needs a live feed, and Google calls add network delay and cost, so cache them and refresh every few minutes.
  - Plan for missing inputs: fall back to the lookup table, then to TransLoc's ETA.
- **Open question:** whether accuracy (more data) or a live product (latency) comes first decides whether the next effort goes into modeling or engineering.

## 6. All models and baselines

15 tuned models (5 per dataset) and 4 baselines per dataset, sorted by test MAE, so each dataset's first row is its best model. Tuning MAE and every number here are also in `summary.csv`.

| Dataset | Family | Model | Test R² | Test MAE (s) | Test RMSE (s) | n_test |
|---|---|---|---|---|---|---|
| bus | lazy | HistGradientBoostingRegressor | 0.504 | 127.4 | 163.5 | 7,536 |
| bus | lazy | ExtraTreesRegressor | 0.386 | 138.7 | 181.9 | 7,536 |
| bus | lazy | LinearSVR | 0.332 | 141.6 | 189.8 | 7,536 |
| bus | xgboost | XGBRegressor | 0.380 | 143.1 | 182.9 | 7,536 |
| bus | nn | MLP | -0.584 | 226.5 | 292.2 | 7,536 |
| bus | baseline | Lookup table (ETA deciles) | 0.021 | 183.4 | 229.7 | 7,536 |
| bus | baseline | Straight line in ETA | 0.002 | 188.5 | 232.0 | 7,536 |
| bus | baseline | Train mean | -0.226 | 218.0 | 257.1 | 7,536 |
| bus | baseline | TransLoc as-is (delay=0) | -1.343 | 278.7 | 355.4 | 7,536 |
| route | lazy | ExtraTreesRegressor | 0.457 | 135.6 | 179.3 | 16,998 |
| route | xgboost | XGBRegressor | 0.449 | 139.2 | 180.6 | 16,998 |
| route | lazy | LGBMRegressor | 0.505 | 139.9 | 171.1 | 16,998 |
| route | lazy | AdaBoostRegressor | 0.360 | 157.7 | 194.7 | 16,998 |
| route | nn | MLP | 0.365 | 158.0 | 193.8 | 16,998 |
| route | baseline | Lookup table (ETA deciles) | 0.206 | 173.0 | 216.8 | 16,998 |
| route | baseline | Straight line in ETA | 0.194 | 176.0 | 218.3 | 16,998 |
| route | baseline | Train mean | -0.087 | 212.2 | 253.6 | 16,998 |
| route | baseline | TransLoc as-is (delay=0) | -1.335 | 290.4 | 371.7 | 16,998 |
| combined | xgboost | XGBRegressor | 0.302 | 142.5 | 232.8 | 29,141 |
| combined | lazy | GradientBoostingRegressor | 0.244 | 156.6 | 242.2 | 29,141 |
| combined | nn | MLP | 0.181 | 176.1 | 252.2 | 29,141 |
| combined | lazy | OrthogonalMatchingPursuit | 0.089 | 191.3 | 265.9 | 29,141 |
| combined | lazy | ElasticNet | 0.136 | 200.7 | 259.0 | 29,141 |
| combined | baseline | Lookup table (ETA deciles) | 0.181 | 178.4 | 252.2 | 29,141 |
| combined | baseline | Straight line in ETA | 0.143 | 193.7 | 258.0 | 29,141 |
| combined | baseline | Train mean | -0.084 | 240.0 | 290.1 | 29,141 |
| combined | baseline | TransLoc as-is (delay=0) | -0.724 | 255.5 | 365.9 | 29,141 |
