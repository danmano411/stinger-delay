# Model comparison: predicting how late TransLoc's ETA is

Three datasets built from the real March 2026 Stinger scrapes, five tuned models on each (15 in total), scored on **held-out service days** with test **R²** and **MAE**.

## Target

`delay_s` = actual arrival − TransLoc's ETA, in seconds. Positive means the bus reached the stop **later** than TransLoc said.

- **Actual arrival:** taken from GPS, as the moment a bus passes within 25 m of the stop's coordinates, interpolated between fixes. Stop coordinates come from `raw_data/route_config/`.
- **Label check:** these arrival times agree with TransLoc's own "stop passed" events for 90% (Gold) and 96% (Green) of stops.

## Datasets (`datasets/`)

| Dataset | Rows (model-ready) | Features | Contents | Train days | Test days |
|---|---|---|---|---|---|
| `bus` | 23,992 | 35 | Green route, bus #3 (the only Green bus seen on two days) | Mar 10 | Mar 9 |
| `route` | 60,809 | 43 | Green route, all 6 buses | Mar 10, 16 | Mar 9 |
| `combined` | 110,399 | 50 | Gold + Green + Red | Mar 3, 5, 6, 10, 16 | Mar 4, 9 |

Pipeline: `src/build_datasets.py` → `<ds>_features.parquet` → `eda/eda_<ds>.ipynb` → `<ds>_model_ready.parquet` (`build_report.md` has every row count).

**Why Clough isn't modeled.** Its scraper never saved which stop an ETA refers to. The only reliable way to recover the stop is to look at where the bus went *next* (92.7% agreement on Red), and that uses the future. The best causal guess, "the stop after the last known arrival", matches TransLoc's own stop only 75.6% of the time on Red, below the 80% bar set in advance. Clough stays in `raw_data/`.

**Cleaning:**
- Drop exact duplicates.
- Drop blank ETAs: on Gold/Green a blank means the bus is at the stop; on Red it means the bus is off its route.
- Drop placeholder rows (no stop and no ETA).
- Drop positions more than 50 m off the route, and overnight rows.
- Drop rows where the bus is already at the target stop.
- Drop **stale ETAs**, where TransLoc still points at the stop the bus just passed: always on Red's next-stop feed, and when the ETA is under 2 min on the all-stop feeds. That ETA refers to the visit that just happened, so the only later arrival is a lap away (a fake ~30 min delay).
- Drop rows whose next arrival is more than 45 min away.
- In EDA: drop target outliers beyond 3×IQR **from the train split only**. Test rows are never filtered by their label.

**Features** (all known at prediction time; the build asserts it):
- **TransLoc ETA:** `eta_s`, its change since the last poll, and how many polls it has been frozen.
- **Trip progress:** stops ahead, TransLoc's planned seconds to the stop, time since the last *known* stop arrival, and distance to the stop. An arrival counts as known only from the GPS fix that reveals it. An out-of-order stop visit that the route-order filter confirms using the next visit counts only from that confirmation.
- **Bus state:** speed, 2-minute mean speed, seconds stationary, stale-GPS flag, position, number of buses on the route.
- **Time:** hour, time-of-day sin/cos, weekday one-hot.
- **Stop:** latitude/longitude, position along the route, planned dwell time.
- **Weather** (Open-Meteo hourly archive, fetched in UTC and converted with real DST rules): temperature, apparent temperature, humidity, precipitation (current and previous hour), cloud cover, wind, gusts, plus a one-hot of the WMO weather group.
- **IDs:** bus and stop IDs become **entity embeddings**, learned by a small PyTorch network on the train split only. Route is one-hot in `combined`.
- **Pruning:** features constant within train are dropped, and for each pair with |Spearman| ≥ 0.95 the one less related to the target is dropped (fit on train).

**Split:**
- Whole service days are held out. Labels reach up to 45 min ahead and delays drift within a day, so any within-day split lets training rows share arrival events and context with test rows.
- The test days are Mar 4 (the first day Gold, Red and Clough were all recorded) and Mar 9 (the first Green day). They were chosen for coverage, before any model was scored on them.
- Every route has both train and test days. Test share is 26–32%.
- Tuning validates the way the test works: on whole held-out days inside train. `combined` uses `GroupKFold(3)` by date and `route` uses `GroupKFold(2)` over its two training days. `bus` has a single training day, so it can only use 3-hour blocks, which reward day-specific quirks (a known weakness).

## Models (`models/<ds>/`)

| File | What it does |
|---|---|
| `lazy_<ds>.ipynb` | LazyRegressor screens ~24 regressors on a group-held-out slice of train, then Optuna tunes the top 3 (grouped 3-fold CV, MAE objective, up to 40 trials or 15 min each) and refits each on the full train set |
| `xgboost_<ds>.ipynb` | XGBRegressor tuned with Optuna (60 trials). Early stopping on an inner validation split; the refit uses the best iteration count |
| `nn_<ds>.py` | PyTorch MLP (LayerNorm, SiLU, dropout) with AdamW, a tuned scheduler (OneCycle or ReduceLROnPlateau), early stopping with best-weight restore, gradient clipping, and Optuna tuning with median pruning. The final model averages the best configuration trained from 5 seeds; each member trains on the 80% of train rows not used for early stopping. Inputs are clipped to the training range |

All files share `src/mc_common.py` for loading, splits, metrics and saving. Each final model writes `results/<ds>__<family>__<model>.json` plus its test predictions in `results/preds/`, so every metric can be recomputed. The metric code checks that y_true in each prediction file matches the dataset exactly.

## Results

Test = held-out service days, each model scored once. MAE is in seconds of error in predicting `delay_s`. Full table with RMSE and tuning scores: `results/summary.md` / `summary.csv`.

| Dataset | Model (family) | Test R² | Test MAE (s) |
|---|---|---|---|
| bus | HistGradientBoostingRegressor (lazy) | **0.504** | **127.4** |
| bus | ExtraTreesRegressor (lazy) | 0.386 | 138.7 |
| bus | XGBRegressor (xgboost) | 0.380 | 143.1 |
| bus | LinearSVR (lazy) | 0.332 | 141.6 |
| bus | MLP (nn) | −0.584 | 226.5 |
| route | LGBMRegressor (lazy) | **0.505** | 139.9 |
| route | ExtraTreesRegressor (lazy) | 0.457 | **135.6** |
| route | XGBRegressor (xgboost) | 0.449 | 139.2 |
| route | MLP (nn) | 0.365 | 158.0 |
| route | AdaBoostRegressor (lazy) | 0.360 | 157.7 |
| combined | XGBRegressor (xgboost) | **0.302** | **142.5** |
| combined | GradientBoostingRegressor (lazy) | 0.244 | 156.6 |
| combined | MLP (nn) | 0.181 | 176.1 |
| combined | ElasticNet (lazy) | 0.136 | 200.7 |
| combined | OrthogonalMatchingPursuit (lazy) | 0.089 | 191.3 |

Baselines on the same test days:

| Dataset | Trust TransLoc (predict delay 0) | Predict the train mean |
|---|---|---|
| bus | R² −1.34, MAE 279 s | R² −0.23, MAE 218 s |
| route | R² −1.34, MAE 290 s | R² −0.09, MAE 212 s |
| combined | R² −0.72, MAE 256 s | R² −0.08, MAE 240 s |

What this says:
- **Tuned models clearly beat trusting TransLoc's ETA as-is.** On a day they have never seen, the best model cuts TransLoc's error by **54% (bus), 53% (route) and 44% (combined)**.
- **Validation has to look like the test.** When `route` was tuned on held-out hours of its training days, the screen promoted linear models that collapsed on the new day (R² −11 to −18), and the NN scored −1.18. Tuned on a held-out *day*, every route model beats the baselines, including the NN (R² 0.365).
- **One training day is too little for a neural network.** `bus` trains on Mar 10 only, and its NN, although correlated with the truth (r = 0.74), predicts about 208 s too high on the calmer test day (mean delay 269 s vs 379 s). With two or more training days the NN's bias shrinks (+77 s route, +52 s combined).
- **Linear models rank poorly on combined** (ElasticNet, OrthogonalMatchingPursuit). The lazy screen validates on one held-out training day, which is a noisy signal, so weak models can reach the top 3.
- **Test sets are one or two days, so gaps of a few hundredths of R² are noise.** Collecting more days (ideally a semester, with all routes recorded at once) would help more than further tuning.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate            # Python 3.12
pip install -r requirements.txt                              # torch: see the note in the file
python -m ipykernel install --user --name stinger-mc
cd src && python build_datasets.py                           # ~30 s
cd ../eda && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=stinger-mc eda_bus.ipynb eda_route.ipynb eda_combined.ipynb
cd ../src && python run_all.py --threads 2                   # 9 jobs in parallel, ~1 h on 16 threads
python summarize_results.py                                  # results/summary.md
```
Set `MC_SMOKE=1` for a fast end-to-end check with tiny budgets.

## Caveats

- **Little data:** about two weeks of March weekdays. Each route was recorded on different days, and the test covers only one or two days per dataset, so scores vary a lot. The `bus` model trains on a single day.
- **Plan times are from later:** planned stop-to-stop times come from the September 2026 route config. The stop IDs match the March data, but the timings may have changed since.
- **Whole-day validation makes early stopping conservative:** on `route`, XGBoost's validation day is rainy Mar 16 against dry Mar 10 for training, so it stopped after 9 trees. That is the honest outcome of validating on a different day; more trees were not tried on the test day.
- **Validation scores are optimistic:** the entity embeddings are fit on all train rows, so tuning/validation scores run higher than test scores. The test split is unaffected.
- **The test days differ from the training days:**
  - Mar 4 is a Wednesday, and no training day is.
  - The Mar 9 test day has no rain, and some of its humidity readings are below anything seen in training.
  - Two of the three Green buses on Mar 9 never appear in training, so they get the average bus embedding.
  - This is what "a new day" looks like with two weeks of data.
- **Heaviest delays are capped out:** rows whose matched arrival is more than 45 min away are removed from both splits, because they are mostly missed-detection matches a lap later. This also removes the largest genuine delays, so absolute errors on extreme days would be higher.
- **Scraper quirks:** see `raw_data/README.md` (ETAs of 0 s saved as blank, Clough stop IDs lost, the GPS-only Gold scraper).

## History (why the method changed)

The first version split rows by service *hour*. An independent verification pass then showed that hour blocks leak: training rows from the hour before a test hour predict the same arrivals. Removing just those rows cut one model's R² from 0.53 to 0.37. The same pass found a small ordering issue in the arrival filter, a one-hour weather offset for dates before Mar 8, and test rows filtered by their label. All of these were fixed, the datasets were rebuilt, and every model was retrained and re-verified. Numbers from the first version are not reported.

A second verification pass then led to two more changes:
1. **Stale ETAs dropped.** Rows where TransLoc still points at the stop the bus just passed were being labeled with the *next lap's* arrival, a fake delay of about +2,000 s. On `combined`, 0.4% of test rows made up 17% of the test error's sum of squares. The rule uses only information available at prediction time, so it applies to both splits.
2. **`route` tuning moved from 3-hour blocks to whole days.** Its lazy screen, validated on held-out hours of the same days, promoted Ridge, BayesianRidge and LinearRegression, which scored R² −11 to −18 on the test day. In training "Monday" is also the rainy, snowy day, so a linear model can pile large opposing weights on those features, and they blow up on a dry Monday. Those results were discarded, and validation now holds out a whole day, like the test. Because this change came after seeing those test scores, it is disclosed here.

The NN recipe changed twice. Both changes are disclosed here:
1. **Seed averaging (first version):** a single retrain of the tuned NN landed 10–25 s of validation MAE away from its tuning trial, so the final fit now averages 5 seeds.
2. **Input clipping (this version):** the first run on held-out days scored R² −0.777 / −1.214 / 0.208 (bus / route / combined). A label-free check showed that test-day features fall outside the training range (for example, 69% of bus test rows have humidity never seen in training). NN inputs are now clipped to the training range, a standard guard against linear extrapolation; tree models are unaffected by construction. With clipping: −0.583 / −1.178 / 0.216. No further changes were made after that, to avoid tuning to the test days.
