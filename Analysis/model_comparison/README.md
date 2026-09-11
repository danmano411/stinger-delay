# Model comparison: correcting TransLoc's Stinger ETAs

This folder predicts how late each TransLoc ETA will be, using the team's Stinger scrapes from March 2026. It builds three datasets, tunes five models on each (15 in total), and scores them on whole days held out from training. On those days the best model for each dataset cuts TransLoc's arrival-time error by 44–54%. Most of that gain comes from correcting TransLoc's steady optimism, which a simple lookup table on its ETA already does.

**Results:** the two key tables are in [section 7](#7-results). Every table, including the worst-case errors and the per-route breakdown, is in **[results/summary.md](results/summary.md)**. Next steps are in [section 9](#9-next-steps).

Pipeline: `raw_data/` → `src/build_datasets.py` → `eda/eda_<ds>.ipynb` → `models/<ds>/` → `src/summarize_results.py` → `results/summary.md`

## 1. Data

- `raw_data/` holds the scrapes: GPS positions and TransLoc ETAs for Gold, Green, Red and Clough (weekdays, Mar 2–16, 2026). It also holds the route configuration (stop coordinates, order, planned times) and hourly weather from Open-Meteo. [raw_data/README.md](raw_data/README.md) lists every file and scraper quirk.
- The three modeled routes have labeled data on seven days, three per route. Green was recorded on different days from Gold and Red, which share Mar 4 and 5. The ~110,000 labeled rows cover only about 4,800 distinct bus-stop arrivals, because every poll repeats the ETA for the same upcoming arrival.
- **Clough is not modeled.** Its scraper never saved which stop an ETA refers to. The best guess that doesn't look into the future ("the stop after the last known arrival") matches TransLoc's own stop only 75.6% of the time on Red, below an 80% bar set in advance.

## 2. Target

`delay_s` = actual arrival − TransLoc's ETA, in seconds. Positive means the bus arrived later than TransLoc said.

The actual arrival is the moment the bus's GPS track passes within 25 m of the stop, interpolated between fixes. An earlier check of this method (not rerun in this pipeline) found these arrivals match TransLoc's own "stop passed" events for 90% (Gold) and 96% (Green) of stops.

## 3. Cleaning

`src/build_datasets.py` drops:
- duplicates, placeholder rows, overnight rows and positions more than 50 m off the route
- blank ETAs (on Gold and Green the bus is at the stop; on Red it's off its route), and rows where the bus is already at the target stop
- **stale ETAs** that still point at the stop the bus just passed: always on Red's next-stop feed, and on the other feeds when the ETA is under 2 min. Their only later arrival is a lap away, which would be a fake ~30 min delay.
- rows whose arrival is more than 45 min away

The EDA notebooks also drop target outliers beyond 3×IQR, from the training split only.

## 4. Features

All are known at prediction time; the build asserts it.
- **TransLoc's ETA:** the ETA, its change since the last poll, and how many polls it has been frozen
- **Trip progress:** stops ahead, TransLoc's planned seconds to the stop, time since the last known stop arrival, distance to the stop
- **Bus state:** speed, 2-minute mean speed, time stationary, stale-GPS flag, position, buses on the route
- **Time:** hour, time-of-day sin/cos, weekday one-hot
- **Stop:** coordinates, position along the route, planned dwell time
- **Weather:** temperature, apparent temperature, humidity, precipitation (this hour and last), cloud cover, wind, gusts, WMO weather-group one-hot
- **IDs:** bus and stop as entity embeddings learned on the training split; route as a one-hot in `combined`

Features constant within train are dropped. For each pair with |Spearman| ≥ 0.95, the one less related to the target is dropped, except that TransLoc's ETA is always kept (fit on train).

## 5. Datasets and split

| Dataset | Contents | Rows (model-ready) | Features | Train days | Test days |
|---|---|---|---|---|---|
| `bus` | Green bus #3, the only Green bus seen on two days | 23,992 | 35 | Mar 10 | Mar 9 |
| `route` | Green, all 6 buses | 60,809 | 43 | Mar 10, 16 | Mar 9 |
| `combined` | Gold + Green + Red | 110,399 | 50 | Mar 3, 5, 6, 10, 16 | Mar 4, 9 |

- **Test = whole held-out days.** Labels reach 45 min ahead and delays drift within a day, so any within-day split leaks. Mar 4 and Mar 9 were chosen for coverage (every route gets a training and a test day) before any model was scored.
- **Tuning validates the same way**, on whole held-out training days. The lazy models use grouped CV by date (3 folds for `combined`, 2 for `route`), and XGBoost and the NN hold out one training day (Mar 5 for `combined`, Mar 16 for `route`). `bus` has one training day, so it validates on 3-hour blocks.
- `eda/eda_<ds>.ipynb` checks outliers, correlations and constants, and writes `datasets/<ds>_model_ready.parquet`. Row counts at every stage are in [datasets/build_report.md](datasets/build_report.md).

## 6. Models

| File | Family | What it does |
|---|---|---|
| `models/<ds>/lazy_<ds>.ipynb` | lazy | LazyRegressor screens ~24 regressors. Optuna tunes the top 3 (grouped CV, MAE objective, up to 40 trials or 15 min each), then each is refit on all of train |
| `models/<ds>/xgboost_<ds>.ipynb` | xgboost | XGBoost tuned with Optuna (60 trials). Early stopping on a held-out part of train picks the number of trees, then it is refit on all of train |
| `models/<ds>/nn_<ds>.py` | nn | PyTorch MLP (LayerNorm, SiLU, dropout, AdamW, OneCycle or ReduceLROnPlateau, early stopping), tuned with Optuna and pruning. The final prediction averages 5 seeds, and inputs are clipped to the training range |

Lazy's top 3 leave out XGBoost, which has its own notebook, so each dataset gets five different models. Every final model saves its test predictions to `results/preds/`, so every metric can be recomputed.

**The NN trains on less data than the others.** Its early stopping holds out part of train (a whole day for `route` and `combined`, one 3-hour block of Mar 10 for `bus`), and the final networks never train on it. That leaves 86% of the training rows for `bus`, 57% for `route` (Mar 10 only) and 62% for `combined`. `combined` holds out Mar 5, which has about 28,000 of Gold's 29,000 training rows. Its NN therefore learned Gold from the 1,002 rows of Mar 6, an unusually calm day.

**Why the NNs stop after 11–32 epochs.** Early stopping ends training once validation error hasn't improved for 10 epochs. [src/nn_epoch_diagnostic.py](src/nn_epoch_diagnostic.py) retrains each tuned NN for 60 epochs without early stopping, on the training split only. It shows they don't stop learning:
- training error keeps falling (`route`: 199 s → 54 s)
- for `route` and `combined`, error on the held-out validation day is lowest at epoch 1–22 (1–9 with the warm-up flaw below fixed), then rises
- validated instead on held-out hours of their own training days (warm-up fixed), the same settings keep improving for 9–57 epochs, down to 98–111 s

So what an NN learns from one day mostly doesn't carry to the next; the limit is the number of days, not epochs. `bus` is the exception: even on a held-out block of its own day, its validation error is lowest at epoch 3–9 and then rises, so with one day of data it overfits quickly.

**A flaw the diagnostic found.** The final fit for `route` and `combined` sized its OneCycle warm-up for 300 epochs (tuning used 150), so the learning rate was only 7.5–31% of its tuned peak when training stopped (`bus` uses a different schedule). With the warm-up sized correctly, validation-day error is still lowest within the first 9 epochs, so the reported NN results were left as they are.

## 7. Results

The two key tables from [results/summary.md](results/summary.md). Test = the held-out days, each model scored once. Error is the mean absolute error of the predicted arrival time, in seconds. **Best model** = the tuned model with the lowest test MAE on that dataset.

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

Error by how far away TransLoc says the bus is:

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

**The models help most from 2 to 20 minutes out**, which is likely when riders decide whether to head to the stop (an assumption; there is no rider data here). In that range the error falls by 48–60% and the share of predictions within 2 minutes of the actual arrival rises by 23–34 points ([summary.md, section 3](results/summary.md#3-how-often-the-prediction-is-within-2-minutes-of-the-actual-arrival)). Under 2 minutes TransLoc is already fairly close (16–36% cut), and beyond 20 minutes (`combined` only) the cut is 29%.

## 8. Caveats

- **Little data.** Each dataset is tested on one or two days, so scores vary a lot, and the best model is picked on those same days.
- **The test days differ from the training days.** Mar 4 is a Wednesday and no training day is. Mar 9 is dry until its last service hour, and some of its humidity readings are below anything in training. Two of the three Green buses on Mar 9 never appear in training, so they get the average bus embedding.
- **Planned stop-to-stop times come from the September 2026 route config.** The stop IDs match March, but the timings may have changed.
- **Tuning scores are not a preview of test scores.** The held-out training days were harder than the test days, so 13 of 15 models scored worse in tuning than on test (`tuning_mae_s` in `summary.csv`). The entity embeddings are learned without the day XGBoost and the NN validate on, but the lazy models' cross-validation folds include days the embeddings were trained on, which flatters those folds slightly. The test split is unaffected.
- **The heaviest delays are capped.** Rows whose arrival is more than 45 min away are dropped from both splits. They are mostly missed detections a lap later, but the cap also removes the largest genuine delays.
- **`route`'s XGBoost has only 9 trees.** Its validation day, Mar 16, was rainy all day (0.5 in), while Mar 10 in training had only light afternoon rain (0.05 in), so early stopping ended it early.

## 9. Next steps

### 9.1 Model structure: `route` or `combined`, decided by the data

- **Drop per-bus models.** A live system needs a prediction for every bus, one bus has the least data, and on bus #3's own rows the pooled model was better ([summary.md §5](results/summary.md#5-by-route-and-pooled-vs-single-route-models)). Bus identity works better as a feature inside a larger model, as the bus embeddings already are.
- **Choose between `route` and `combined` from the data we collect.** Today `combined` wins on Green, because pooling adds days no single route has. If each route gets many days of its own, per-route models may win where routes behave differently (Red's feed, for instance, only gives next-stop ETAs). Decide per route: train both on the same days, evaluate day by day (train on everything before a day, test on that day, repeat), and keep whichever wins. A middle path is one pooled model with a per-route correction on top.

### 9.2 More information to collect

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

### 9.3 Which model, and what would change it

- **Now:** a gradient-boosted tree model (XGBoost, LightGBM or HistGradientBoosting). Tree ensembles won on every dataset, and a gradient-boosted model was best or within 3% of the best ([summary.md §6](results/summary.md#6-all-models-and-baselines)). They train in minutes and handle mixed features well. Ship a refined lookup table first (9.4) as the simplest win, and keep it as the fallback when a feature feed is down. Shelve the NN.
- **If we get much more data** (months, all routes):
  - NNs become worth revisiting, including sequence models that read a bus's recent GPS trace.
  - Per-route models may start to win (9.1).
  - Prediction ranges (9.4) become trustworthy, because there are enough days to check them.
  - Retrain on a schedule, for example weekly, with day-by-day evaluation.
- **If it has to run live at low latency** (a prediction for every bus and stop on every poll, every 15–30 s):
  - The lookup table is instant, and a gradient-boosted model takes milliseconds per batch; cap its tree count and depth. Large ExtraTrees forests and 5-seed NN ensembles cost more.
  - Features, not models, are the bottleneck. Rolling features (2-minute speed, time since the last arrival) need live per-bus state, weather needs a live feed, and Google calls add network delay and cost, so cache them and refresh every few minutes.
  - Plan for missing inputs: fall back to the lookup table, then to TransLoc's ETA.
- **Open question:** whether accuracy (more data) or a live product (latency) comes first decides whether the next effort goes into modeling or engineering.

### 9.4 Uncertainty: the worst predictions matter more than the average

Being more accurate 95% of the time doesn't help if the other 5% is outrageous: one prediction that is 15 minutes off costs more rider trust than many small wins earn. [summary.md §4](results/summary.md#4-the-worst-predictions) shows the tails:
- **The tails shrink, but not everywhere.** Predictions more than 10 min off fall from 9.7% and 11.5% for TransLoc to 0.2% for the best `bus` and `route` models. On `combined`, though, the model's worst 1% are worse than the lookup table's (99th percentile 882 s vs 765 s).
- **Direction matters.** If the bus comes earlier than predicted, a rider who trusts the prediction misses it; if later, they wait. TransLoc's optimism almost never errs early (0.0–0.2% of its predictions are more than 5 min early). The corrections do: 3.8–7.6% for the best models, and 9.6–15.4% for the lookup table, which adds the average delay even to buses running on time.

What to do:
1. **Predict a range, not a single time.** Quantile versions of the same models (XGBoost, LightGBM and HistGradientBoosting all offer a quantile loss) give, say, the 10th and 90th percentile arrival. Calibrating the range on held-out days (conformal prediction) makes a "90% range" contain the real arrival 90% of the time.
2. **Lean early.** Show riders the early end of the range, or train for a lower percentile, so misses turn into waits instead of missed buses.
3. **Track the tails, not just the average.** For every test day, report the 95th and 99th percentile error and the share of predictions more than 5 min early, by route and distance.
4. **Add guardrails.** The outrageous 5% should stay manageable if the lookup table is refined and paired with a deterministic flag, shown alongside the bus's actual location that riders already see and use:
   - Refine the lookup table by route, stop, distance and time of day, and have it add a cautious percentile of past delays rather than the mean.
   - Flag predictions that disagree sharply with the deterministic baseline (9.2) or are physically impossible, such as "2 min away" while the bus is 3 km out. Clamp them or fall back to the lookup table.
   - Riders already judge ETAs against the bus's live position on the map, so a flag shown there lets them spot an outlier instead of trusting it.

## 10. How the method changed

Independent verification passes changed the method several times. Numbers from earlier versions are quoted only to explain each change. In order:
1. **NN seed averaging.** A single retrain of a tuned NN varied by 10–25 s of validation MAE, so the final fit averages 5 seeds.
2. **Hour-based split → whole days.** The first version held out service hours, and training rows from the hour before a test hour predicted the same arrivals. Removing them cut one model's R² from 0.53 to 0.37. The same pass fixed an ordering issue in the arrival filter, a one-hour weather offset before Mar 8, and test rows filtered by their own label.
3. **NN inputs clipped to the training range,** because test-day features fell outside it. In that run, NN test R² went from −0.777 / −1.214 / 0.208 (`bus` / `route` / `combined`) to −0.583 / −1.178 / 0.216. This change was made after seeing those test scores.
4. **Stale ETAs dropped.** They were labeled with the next lap's arrival (about +2,000 s). On `combined`, 0.4% of test rows made up 17% of the squared error.
5. **`route` tuning moved from 3-hour blocks to whole days.** Validated within the day, its screen picked Ridge, BayesianRidge and LinearRegression, which scored R² −11 to −18 on the test day. This change was also made after seeing test scores. It was the last change: nothing was changed after it to chase test scores.

## 11. Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate            # Python 3.12
pip install -r requirements.txt                              # torch: see the note in the file
python -m ipykernel install --user --name stinger-mc
cd src && python build_datasets.py                           # ~30 s
cd ../eda && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=stinger-mc eda_bus.ipynb eda_route.ipynb eda_combined.ipynb
cd ../src && python run_all.py --threads 2                   # 9 jobs in parallel, ~1 h on 16 threads
python summarize_results.py                                  # results/summary.md and summary.csv
python nn_epoch_diagnostic.py route 4                        # optional: the NN epoch diagnostic (also bus, combined)
```
Set `MC_SMOKE=1` for a fast end-to-end check with tiny budgets.
