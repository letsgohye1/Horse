# Horse Rank Model Summary

## Why this approach

- PDF 1 suggested treating ordered performance as a regression problem and checking `MAE`, `RMSE`, and `R2`.
- PDF 3 emphasized `Pipeline`, validation split, and boosting models, so the main model was set to `LightGBM`.
- PDF 2 classification metrics were treated as secondary because race finish order is ordinal rather than a plain category label.
- PDF 4 unsupervised learning and dimensionality reduction were not used in this first version because the feature count is small and interpretability matters more here.

## Target setup

- Main target: `순위점수`
- Reason: raw `순위` is not directly comparable across races with different field sizes, while `순위점수` is normalized by race context.
- Leakage removed: `경주기록`, `순위`, `순위점수`

## Feature work

- Parsed race date and birth date
- Added:
  - `말나이_일`
  - `말나이_년`
  - `경주연도`
  - `경주월`
  - `경주요일`
- Categorical features were one-hot encoded
- Numeric features were median-imputed

## Data split

- Chronological split by `경주일자`
- Train: 7,014 rows
- Validation: 3,483 rows
- Test: 4,963 rows

## Final test metrics

- `MAE`: 0.2521
- `RMSE`: 0.2971
- `R2`: 0.0949
- `winner_accuracy`: 0.2819
- `winner_in_top_3_rate`: 0.5752
- `ndcg_at_3`: 0.7260

## Artifacts

- Model: `rank_model_lightgbm.joblib`
- Metrics: `rank_model_results.json`
- Test predictions: `rank_model_test_predictions.csv`
- PDF notes: `lecture_pdf_notes.txt`
