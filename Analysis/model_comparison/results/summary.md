# Model comparison results

15 tuned models. Target `delay_s` (seconds the bus arrived later than TransLoc's ETA); test split = held-out service days (Mar 4, Mar 9), scored once per model.

| Dataset | Family | Model | Test R² | Test MAE (s) | Test RMSE (s) | n_test |
|---|---|---|---|---|---|---|
| bus | lazy | HistGradientBoostingRegressor | 0.504 | 127.4 | 163.5 | 7,536 |
| bus | lazy | ExtraTreesRegressor | 0.386 | 138.7 | 181.9 | 7,536 |
| bus | xgboost | XGBRegressor | 0.380 | 143.1 | 182.9 | 7,536 |
| bus | lazy | LinearSVR | 0.332 | 141.6 | 189.8 | 7,536 |
| bus | nn | MLP | -0.584 | 226.5 | 292.2 | 7,536 |
| bus | baseline | Train mean | -0.226 | 218.0 | 257.1 | 7,536 |
| bus | baseline | TransLoc as-is (delay=0) | -1.343 | 278.7 | 355.4 | 7,536 |
| route | lazy | LGBMRegressor | 0.505 | 139.9 | 171.1 | 16,998 |
| route | lazy | ExtraTreesRegressor | 0.457 | 135.6 | 179.3 | 16,998 |
| route | xgboost | XGBRegressor | 0.449 | 139.2 | 180.6 | 16,998 |
| route | nn | MLP | 0.365 | 158.0 | 193.8 | 16,998 |
| route | lazy | AdaBoostRegressor | 0.360 | 157.7 | 194.7 | 16,998 |
| route | baseline | Train mean | -0.087 | 212.2 | 253.6 | 16,998 |
| route | baseline | TransLoc as-is (delay=0) | -1.335 | 290.4 | 371.7 | 16,998 |
| combined | xgboost | XGBRegressor | 0.302 | 142.5 | 232.8 | 29,141 |
| combined | lazy | GradientBoostingRegressor | 0.244 | 156.6 | 242.2 | 29,141 |
| combined | nn | MLP | 0.181 | 176.1 | 252.2 | 29,141 |
| combined | lazy | ElasticNet | 0.136 | 200.7 | 259.0 | 29,141 |
| combined | lazy | OrthogonalMatchingPursuit | 0.089 | 191.3 | 265.9 | 29,141 |
| combined | baseline | Train mean | -0.084 | 240.0 | 290.1 | 29,141 |
| combined | baseline | TransLoc as-is (delay=0) | -0.724 | 255.5 | 365.9 | 29,141 |
