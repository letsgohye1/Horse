# Horse Rank Model with Advanced Features

## Added feature groups

- Same horse recent N-race trend
  - `horse_score_lag_1/2/3`
  - `horse_rank_lag_1/2/3`
  - `horse_score_mean_last_2/3/5`
  - `horse_rank_mean_last_2/3/5`
  - `horse_score_trend_1_vs_3`
  - `horse_score_trend_1_vs_5`
- Horse-by-distance history
  - `horse_distance_prev_count`
  - `horse_distance_prev_score_mean`
  - `horse_distance_prev_top3_rate`
  - `horse_distance_last_score`
- Horse-by-grade history
  - `horse_grade_ctx_prev_count`
  - `horse_grade_ctx_prev_score_mean`
  - `horse_grade_ctx_prev_win_rate`
  - `horse_grade_ctx_last_rank`
- Horse-jockey combo history
  - `horse_jockey_combo_prev_count`
  - `horse_jockey_combo_prev_score_mean`
  - `horse_jockey_combo_prev_win_rate`
  - `horse_jockey_combo_last_score`

## Test comparison

### Baseline

- `MAE`: 0.2521
- `RMSE`: 0.2971
- `R2`: 0.0949
- `winner_accuracy`: 0.2819
- `winner_in_top_3_rate`: 0.5752
- `ndcg_at_3`: 0.7260

### History features

- `MAE`: 0.2431
- `RMSE`: 0.2886
- `R2`: 0.1455
- `winner_accuracy`: 0.2895
- `winner_in_top_3_rate`: 0.6229
- `ndcg_at_3`: 0.7505

### Advanced features

- `MAE`: 0.2415
- `RMSE`: 0.2868
- `R2`: 0.1567
- `winner_accuracy`: 0.2781
- `winner_in_top_3_rate`: 0.6457
- `ndcg_at_3`: 0.7534

## Read on the result

- Regression quality improved again.
- Ranking quality near the top also improved.
- Exact winner hit rate dropped a bit compared with the history-only version.
- So this version is stronger if your goal is better overall ordering or top-3 candidate quality.

## Main files

- Notebook: `rank_model_advanced_features.ipynb`
- Script: `train_rank_model_advanced.py`
- Model: `rank_model_advanced_lightgbm.joblib`
- Predictions: `rank_model_advanced_test_predictions.csv`
- Feature importance: `rank_model_advanced_feature_importance.csv`
- Metrics: `rank_model_advanced_results.json`
