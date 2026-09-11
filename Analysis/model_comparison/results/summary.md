# Model comparison results

## Quick scan

Test = held-out service days (Mar 4, Mar 9). Error = mean absolute error of the predicted arrival time, in seconds. Predicting `delay_s` is the same as correcting TransLoc's ETA, so every row is comparable.

| Average error (MAE) | bus | route | combined |
|---|---|---|---|
| TransLoc as-is (the current system) | 279 s | 290 s | 256 s |
| Add the average training delay to every ETA | 218 s | 212 s | 240 s |
| Straight-line fix: delay = a + b × ETA | 188 s | 176 s | 194 s |
| Lookup table: average training delay per tenth of the ETA range | 183 s | 173 s | 178 s |
| **Best tuned model** (lowest test MAE) | **127 s** HistGradientBoostingRegressor | **136 s** ExtraTreesRegressor | **143 s** XGBRegressor |

The ETA fixes and lookup tables are fit on train, like the models. The better of the two gets 63%–76% of the best model's improvement over TransLoc. The best model then cuts the remaining error by another 20%–31%.

Error by how far away TransLoc says the bus is (ranges across the datasets):

| TransLoc's ETA | TransLoc's error | Best model's error | Bus arrives later than TransLoc said, on average |
|---|---|---|---|
| under 2 min | 88–103 s | 66–83 s | +1.4 to +1.7 min |
| 2–5 min | 178–208 s | 92–106 s | +2.8 to +3.4 min |
| 5–10 min | 308–340 s | 134–154 s | +4.7 to +5.3 min |
| 10–20 min | 418–438 s | 169–214 s | +6.7 to +7.3 min |
| 20+ min (combined only) | 463 s | 328 s | +6.5 min |

Within 2 min of the actual arrival: TransLoc 29%–41% of predictions, best model 55%–63%.

## All models and baselines

15 tuned models. Target `delay_s` (seconds the bus arrived later than TransLoc's ETA); test split = held-out service days (Mar 4, Mar 9), scored once per model.

| Dataset | Family | Model | Test R² | Test MAE (s) | Test RMSE (s) | n_test |
|---|---|---|---|---|---|---|
| bus | lazy | HistGradientBoostingRegressor | 0.504 | 127.4 | 163.5 | 7,536 |
| bus | lazy | ExtraTreesRegressor | 0.386 | 138.7 | 181.9 | 7,536 |
| bus | xgboost | XGBRegressor | 0.380 | 143.1 | 182.9 | 7,536 |
| bus | lazy | LinearSVR | 0.332 | 141.6 | 189.8 | 7,536 |
| bus | nn | MLP | -0.584 | 226.5 | 292.2 | 7,536 |
| bus | baseline | ETA-decile table | 0.021 | 183.4 | 229.7 | 7,536 |
| bus | baseline | Line in ETA | 0.002 | 188.5 | 232.0 | 7,536 |
| bus | baseline | Train mean | -0.226 | 218.0 | 257.1 | 7,536 |
| bus | baseline | TransLoc as-is (delay=0) | -1.343 | 278.7 | 355.4 | 7,536 |
| route | lazy | LGBMRegressor | 0.505 | 139.9 | 171.1 | 16,998 |
| route | lazy | ExtraTreesRegressor | 0.457 | 135.6 | 179.3 | 16,998 |
| route | xgboost | XGBRegressor | 0.449 | 139.2 | 180.6 | 16,998 |
| route | nn | MLP | 0.365 | 158.0 | 193.8 | 16,998 |
| route | lazy | AdaBoostRegressor | 0.360 | 157.7 | 194.7 | 16,998 |
| route | baseline | ETA-decile table | 0.206 | 173.0 | 216.8 | 16,998 |
| route | baseline | Line in ETA | 0.194 | 176.0 | 218.3 | 16,998 |
| route | baseline | Train mean | -0.087 | 212.2 | 253.6 | 16,998 |
| route | baseline | TransLoc as-is (delay=0) | -1.335 | 290.4 | 371.7 | 16,998 |
| combined | xgboost | XGBRegressor | 0.302 | 142.5 | 232.8 | 29,141 |
| combined | lazy | GradientBoostingRegressor | 0.244 | 156.6 | 242.2 | 29,141 |
| combined | nn | MLP | 0.181 | 176.1 | 252.2 | 29,141 |
| combined | lazy | ElasticNet | 0.136 | 200.7 | 259.0 | 29,141 |
| combined | lazy | OrthogonalMatchingPursuit | 0.089 | 191.3 | 265.9 | 29,141 |
| combined | baseline | ETA-decile table | 0.181 | 178.4 | 252.2 | 29,141 |
| combined | baseline | Line in ETA | 0.143 | 193.7 | 258.0 | 29,141 |
| combined | baseline | Train mean | -0.084 | 240.0 | 290.1 | 29,141 |
| combined | baseline | TransLoc as-is (delay=0) | -0.724 | 255.5 | 365.9 | 29,141 |
