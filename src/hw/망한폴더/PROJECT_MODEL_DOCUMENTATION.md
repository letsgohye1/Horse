# Horse Ranking Model Documentation

## 1. Overview

This folder contains three generations of horse-race ranking models built from:

- source data: `C:\Users\Admin\Desktop\hhh\Horse\data_preprocessing\merged_data_kr_Nan.csv`
- output location restriction: `C:\Users\Admin\Desktop\hhh\Horse\src\hw`

The models predict the normalized race outcome target stored in the dataset as `순위점수`.

The project evolved in three stages:

1. `Baseline`
   - current-race information only
   - simple date-derived features
2. `History features`
   - adds horse / jockey / trainer past-performance summaries
3. `Advanced features`
   - adds recent trend features
   - adds horse-by-distance / horse-by-grade context history
   - adds horse-jockey combination history

The project also includes:

- saved model artifacts (`.joblib`)
- prediction outputs (`.csv`)
- metric summaries (`.json`)
- runnable Jupyter notebooks (`.ipynb`)
- split comparison references between different evaluation protocols


## 2. Main Files

### Training scripts

- [train_rank_model.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model.py:1)
  - baseline model
- [train_rank_model_history.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model_history.py:1)
  - history-feature model
- [train_rank_model_advanced.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model_advanced.py:1)
  - advanced-feature model

### Loading helper

- [load_saved_models.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/load_saved_models.py:1)
  - quick example for `joblib.load()`

### Notebooks

- [rank_model_history_features.ipynb](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_history_features.ipynb)
  - trains and reviews history-feature model
- [rank_model_advanced_features.ipynb](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_advanced_features.ipynb)
  - trains and reviews advanced-feature model
- [rank_model_compare_2026_holdout.ipynb](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_compare_2026_holdout.ipynb)
  - compares the three models under the 2026 holdout split
- [split_comparison.ipynb](C:/Users/Admin/Desktop/hhh/Horse/src/hw/split_comparison.ipynb)
  - compares `70/15/15` results against `2026 holdout` results

### Summaries and references

- [rank_model_summary.md](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_summary.md:1)
- [rank_model_history_summary.md](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_history_summary.md:1)
- [rank_model_advanced_summary.md](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_advanced_summary.md:1)
- [split_comparison_summary.md](C:/Users/Admin/Desktop/hhh/Horse/src/hw/split_comparison_summary.md:1)
- [split_comparison_reference.json](C:/Users/Admin/Desktop/hhh/Horse/src/hw/split_comparison_reference.json)


## 3. Target and Problem Formulation

### Prediction target

The models use:

- target: `순위점수`

instead of:

- raw finish rank `순위`

### Why `순위점수` was chosen

Raw rank is hard to compare across races because field sizes differ.  
For example, finishing 3rd in a 12-horse race is not the same as finishing 3rd in a 5-horse race.

`순위점수` is a normalized score that makes cross-race learning more stable.

### Leakage prevention

The following columns are intentionally removed before training:

- finish time / race result features already determined after the race
- rank
- normalized rank score itself

Specifically:

- baseline/history scripts drop `경주기록`, `순위`, `순위점수`
- advanced script drops the corresponding internally mapped columns `race_time`, `rank`, `score`


## 4. Data Flow

All three scripts follow the same broad pipeline:

1. load CSV
2. parse date columns
3. construct `race_id`
4. build features
5. split into train / validation / test
6. preprocess numeric and categorical columns
7. train Ridge baseline
8. tune LightGBM on validation set
9. retrain final LightGBM on `train + validation`
10. evaluate on test set
11. save model, metrics, predictions, and sometimes feature importance


## 5. Shared Modeling Structure

### 5.1 Preprocessing

All models use a `ColumnTransformer`:

- numeric columns:
  - `SimpleImputer(strategy="median")`
- categorical columns:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore", min_frequency=5)`

### 5.2 Models

Each training script evaluates:

- `Ridge`
  - simple regression baseline
- `LightGBMRegressor`
  - main ranking-score regressor

### 5.3 LightGBM parameter search

The scripts use a small manual candidate grid rather than full `GridSearchCV` or Optuna.

Typical candidates vary:

- `n_estimators`
- `learning_rate`
- `num_leaves`
- `min_child_samples`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`

The best candidate is chosen using validation `RMSE`.


## 6. Evaluation Metrics

The models report both regression metrics and race-level ranking metrics.

### Regression metrics

- `MAE`
- `RMSE`
- `R2`

### Race-level ranking metrics

These are computed by grouping rows with the same `race_id`.

- `winner_accuracy`
  - whether the top predicted horse exactly matches the actual winner
- `winner_in_top_3_rate`
  - whether the actual winner appears in the model's top 3 predicted horses
- `ndcg_at_3`
  - ranking-quality score over top 3 positions using the actual normalized score as relevance

The ranking-metric logic is implemented in each script's `race_level_metrics()` function.


## 7. Split Strategies Used in This Project

Two split strategies were used during the project.

### 7.1 Initial split: chronological 70 / 15 / 15

This was the original setup.

- all race dates were sorted
- earliest 70% of dates -> train
- next 15% -> validation
- final 15% -> test

These numbers are preserved in:

- [split_comparison_reference.json](C:/Users/Admin/Desktop/hhh/Horse/src/hw/split_comparison_reference.json)

### 7.2 Current split: 2026 holdout

This is the current active setup in the training scripts.

- `train`: dates before `2026-01-01`
- `validation`: first half of 2026 race dates
- `test`: second half of 2026 race dates

This is implemented in:

- [train_rank_model.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model.py:63)
- [train_rank_model_history.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model_history.py:117)
- [train_rank_model_advanced.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model_advanced.py:219)

### Why the metrics changed

When the split changed, the evaluation set also changed.  
So scores from the original `70/15/15` split and the newer `2026 holdout` split should not be compared as if they came from the same test distribution.


## 8. Baseline Model

### File

- [train_rank_model.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model.py:1)

### Feature design

Baseline uses mostly current-race columns plus a few date-derived features:

- age in days
- age in years
- race year
- race month
- race weekday

### Key functions

- `load_data()`
  - reads CSV, parses dates, builds `race_id`
- `add_features()`
  - creates the simple date-derived features
- `split_by_date()`
  - applies the current `2026 holdout` split
- `build_preprocessor()`
  - numeric and categorical preprocessing
- `tune_lightgbm()`
  - selects best candidate on validation RMSE
- `save_predictions()`
  - writes prediction CSV

### Artifacts

- `rank_model_lightgbm.joblib`
- `rank_model_results.json`
- `rank_model_test_predictions.csv`


## 9. History Features Model

### File

- [train_rank_model_history.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model_history.py:1)

### Core idea

This version adds cumulative past-performance features for:

- horse
- jockey
- trainer

### Entity history features

For each entity, the script builds:

- previous start count
- previous mean normalized score
- previous mean rank
- previous mean horse weight
- previous win count
- previous top-3 count
- previous win rate
- previous top-3 rate
- last race score
- last race rank

### Important implementation notes

- history is computed in date order
- features only use prior rows for the same entity
- current-row target information is excluded from history through subtraction / shifting

### Key functions

- `add_base_features()`
- `add_history_features()`
- `_history_prefix()`
- `run_training()`

### Artifacts

- `rank_model_history_lightgbm.joblib`
- `rank_model_history_results.json`
- `rank_model_history_test_predictions.csv`
- `rank_model_history_feature_importance.csv`


## 10. Advanced Features Model

### File

- [train_rank_model_advanced.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/train_rank_model_advanced.py:1)

### Why this script is structurally different

This script maps dataset columns to internal English aliases using `_resolve_columns()`.

That was done to make the code more stable and readable while working with Korean column names and shell encoding issues.

### Feature groups

#### 10.1 Base features

- age in days
- age in years
- race year
- race month
- race weekday

#### 10.2 Horse / jockey / trainer history

Same general idea as the history model, but implemented using reusable helper functions:

- `_add_entity_history_features()`

#### 10.3 Recent horse trend features

Implemented by:

- `_add_recent_trend_features()`

Created features include:

- `horse_score_lag_1/2/3`
- `horse_rank_lag_1/2/3`
- `horse_score_mean_last_2/3/5`
- `horse_rank_mean_last_2/3/5`
- `horse_score_trend_1_vs_3`
- `horse_rank_trend_1_vs_3`
- `horse_score_trend_1_vs_5`

#### 10.4 Context-specific history

Implemented by:

- `_add_context_history_features()`

Used for:

- horse + distance
- horse + race grade
- horse + jockey combination

Examples:

- `horse_distance_prev_score_mean`
- `horse_grade_ctx_prev_top3_rate`
- `horse_jockey_combo_prev_count`
- `horse_jockey_combo_last_score`

### Key functions

- `_resolve_columns()`
  - maps original CSV columns to internal English names by column position
- `add_advanced_history_features()`
  - orchestrates all advanced feature creation
- `extract_feature_importance()`
  - saves LightGBM feature importance

### Artifacts

- `rank_model_advanced_lightgbm.joblib`
- `rank_model_advanced_results.json`
- `rank_model_advanced_test_predictions.csv`
- `rank_model_advanced_feature_importance.csv`


## 11. Saved Model Files

### Current saved models

- [rank_model_lightgbm.joblib](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_lightgbm.joblib)
- [rank_model_history_lightgbm.joblib](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_history_lightgbm.joblib)
- [rank_model_advanced_lightgbm.joblib](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_advanced_lightgbm.joblib)

### Loading example

Use either:

- [load_saved_models.py](C:/Users/Admin/Desktop/hhh/Horse/src/hw/load_saved_models.py:1)

or directly:

```python
import joblib

baseline_model = joblib.load(r"C:\Users\Admin\Desktop\hhh\Horse\src\hw\rank_model_lightgbm.joblib")
history_model = joblib.load(r"C:\Users\Admin\Desktop\hhh\Horse\src\hw\rank_model_history_lightgbm.joblib")
advanced_model = joblib.load(r"C:\Users\Admin\Desktop\hhh\Horse\src\hw\rank_model_advanced_lightgbm.joblib")
```


## 12. Notebooks and Their Purpose

### `rank_model_history_features.ipynb`

Purpose:

- train history model
- save outputs
- display metrics
- compare against baseline result file

### `rank_model_advanced_features.ipynb`

Purpose:

- train advanced model
- save outputs
- display top feature importance
- compare baseline / history / advanced

### `rank_model_compare_2026_holdout.ipynb`

Purpose:

- compare the three current models under the same active `2026 holdout` split

### `split_comparison.ipynb`

Purpose:

- compare results from:
  - original `70/15/15` split
  - current `2026 holdout` split


## 13. How to Re-run

### Baseline

```powershell
C:\Users\Admin\Desktop\hhh\Horse\.venv\Scripts\python.exe .\src\hw\train_rank_model.py
```

### History model

```powershell
C:\Users\Admin\Desktop\hhh\Horse\.venv\Scripts\python.exe .\src\hw\train_rank_model_history.py
```

### Advanced model

```powershell
C:\Users\Admin\Desktop\hhh\Horse\.venv\Scripts\python.exe .\src\hw\train_rank_model_advanced.py
```

### Execute notebooks non-interactively

```powershell
C:\Users\Admin\Desktop\hhh\Horse\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace .\src\hw\rank_model_history_features.ipynb
C:\Users\Admin\Desktop\hhh\Horse\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace .\src\hw\rank_model_advanced_features.ipynb
C:\Users\Admin\Desktop\hhh\Horse\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace .\src\hw\rank_model_compare_2026_holdout.ipynb
C:\Users\Admin\Desktop\hhh\Horse\.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace .\src\hw\split_comparison.ipynb
```


## 14. Current Interpretation of Model Results

### Under the current `2026 holdout` split

The latest comparison shows:

- `Baseline`
  - weakest overall
- `History features`
  - strong improvement over baseline
  - slightly better `winner_in_top_3_rate` than advanced
- `Advanced features`
  - best `MAE`, `RMSE`, `R2`, `winner_accuracy`, `ndcg_at_3`
  - not always the top score for `winner_in_top_3_rate`

So the practical reading is:

- use `advanced` when overall ranking quality is the main goal
- consider `history` if top-3 winner coverage is the most important single metric


## 15. Known Caveats

### 15.1 Encoding noise in terminal output

Some Korean column names appear broken in terminal output or raw script dumps.  
This is mainly a shell display issue. The scripts themselves read the CSV correctly using `utf-8-sig`.

The advanced script is the cleanest implementation in this regard because it uses internal English aliases.

### 15.2 Result files were overwritten during iteration

The current `rank_model_results.json`, `rank_model_history_results.json`, and `rank_model_advanced_results.json` reflect the latest `2026 holdout` training runs.

The original `70/15/15` summary values are preserved in:

- [split_comparison_reference.json](C:/Users/Admin/Desktop/hhh/Horse/src/hw/split_comparison_reference.json)

### 15.3 Manual tuning grid

The hyperparameter search is intentionally compact.  
This keeps iteration fast, but it is not an exhaustive search.


## 16. Recommended Next Steps

If the project continues, the most promising next upgrades are:

1. add rolling validation instead of a single holdout split
2. create a separate winner-classification auxiliary model
3. build explicit race-day inference preprocessing from raw upcoming-entry data
4. add horse-jockey-distance and horse-grade-distance interaction histories
5. try Optuna or a wider LightGBM search space
6. export a single production inference notebook or script


## 17. Quick Reference

If you only need the essentials:

- best all-around current model:
  - [rank_model_advanced_lightgbm.joblib](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_advanced_lightgbm.joblib)
- best notebook for current workflow:
  - [rank_model_advanced_features.ipynb](C:/Users/Admin/Desktop/hhh/Horse/src/hw/rank_model_advanced_features.ipynb)
- best notebook for understanding split differences:
  - [split_comparison.ipynb](C:/Users/Admin/Desktop/hhh/Horse/src/hw/split_comparison.ipynb)
