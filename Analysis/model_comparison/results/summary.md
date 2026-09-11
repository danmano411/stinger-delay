# Results

Test days are held out whole (Mar 4 and Mar 9), and each model is scored on them once. Error is the mean absolute error (MAE) of the predicted arrival time, in seconds. Predicting `delay_s` is the same as correcting TransLoc's ETA, so "TransLoc as-is" is simply a prediction of zero delay.

**Best model** means the tuned model with the lowest test MAE on that dataset: HistGradientBoostingRegressor for `bus`, ExtraTreesRegressor for `route`, XGBRegressor for `combined`. It is picked on the test days, so its score is slightly optimistic. Percentages are computed from the rounded numbers shown.

## 1. Best model vs TransLoc, and where the gain comes from

The lookup table is the simplest possible fix: it uses nothing but TransLoc's ETA, splits it into 10 ranges, and adds the average training delay for that range. The best model adds the rest using the other features.

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

1. **Collect more days, with all routes recorded on the same days.** The 110,569 labeled rows are only about 4,800 distinct bus-stop arrivals over 7 days, and each route has two training days. Days differ a lot (Gold's average delay was 18 s on Mar 6 and 356 s on Mar 5), and what a model learns from one day mostly doesn't carry to the next (README section 6). More days also tighten the error bars: with one test day per route there is no honest interval yet, and it narrows with the square root of the number of days.
2. **Ask GT Parking & Transportation for historical data.** The public API has no history, but the RideSystems dispatch side may keep it. Months of data at once would beat a semester of scraping.
3. **Fix the scrapers and save more fields.** Recover Clough's stop ID and ETAs of 0 s (saved as blank), and start saving occupancy, IsDelayed, OnTimeStatus and heading.
4. **Keep the pooled model, and test per-route models as data grows.** With sparse data it seems natural to fit each route on its own, but on Green pooling already won (section 4). Gold and Red never got their own models, so train per-route and pooled models on the same days and compare them on the same rows. A pooled model with route and stop embeddings is also the one that can learn what routes share, such as the campus road layout, traffic and weather.
5. **Evaluate day by day.** Train on everything before a day, test on that day, and repeat. This gives a per-day error distribution with real confidence intervals, by route and by distance.
6. **Ship the lookup table now.** It uses only TransLoc's ETA, is easy to explain, and gets most of the gain (section 1). Refit it as data comes in, keep gradient boosting as the upgrade, and shelve the NN until there are dozens of days.
7. **Add features that explain bad days:** GT class-change times, game days and other events, occupancy, and the gap to the bus ahead.

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
