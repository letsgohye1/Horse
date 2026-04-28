# Horse Rank Model with History Features

## What changed

- Added cumulative historical features for:
  - horse (`마명`)
  - jockey (`기수번호`)
  - trainer (`조교사번호`)
- Each entity now contributes features such as:
  - previous race count
  - previous mean score
  - previous mean rank
  - previous win rate
  - previous top-3 rate
  - last race score
  - last race rank

## Why this helped

- The earlier model mostly used current-race context.
- The new version gives the model each horse/jockey/trainer's form entering the race.
- This is usually much closer to the real signal for ranking tasks.

## Test performance

### Previous model

- `MAE`: 0.2521
- `RMSE`: 0.2971
- `R2`: 0.0949
- `winner_accuracy`: 0.2819
- `winner_in_top_3_rate`: 0.5752
- `ndcg_at_3`: 0.7260

### History-feature model

- `MAE`: 0.2431
- `RMSE`: 0.2886
- `R2`: 0.1455
- `winner_accuracy`: 0.2895
- `winner_in_top_3_rate`: 0.6229
- `ndcg_at_3`: 0.7505

## Main artifacts

- Notebook: `rank_model_history_features.ipynb`
- Training script: `train_rank_model_history.py`
- Model: `rank_model_history_lightgbm.joblib`
- Predictions: `rank_model_history_test_predictions.csv`
- Feature importance: `rank_model_history_feature_importance.csv`
- Metrics: `rank_model_history_results.json`

## Notebook usage

Open `rank_model_history_features.ipynb` and run all cells.

The notebook will:

1. train the history-feature model
2. save the model and prediction files
3. show test metrics
4. show top feature importance
5. compare the new model against the previous baseline
