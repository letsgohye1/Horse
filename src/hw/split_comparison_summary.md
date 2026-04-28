# Split Comparison Summary

## Why the numbers changed

The model family did not suddenly get worse. The evaluation split changed.

- `70/15/15`:
  - chronological split across the full timeline
  - train/validation/test all come from the broader 2023-2026 period
- `2026 holdout`:
  - train uses only dates before `2026-01-01`
  - validation/test use only 2026 dates
  - this is a harder and more realistic future-only evaluation

## 70/15/15 results

| model | mae | rmse | r2 | winner_accuracy | winner_in_top_3_rate | ndcg_at_3 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.2521 | 0.2971 | 0.0949 | 0.2819 | 0.5752 | 0.7260 |
| history_features | 0.2431 | 0.2886 | 0.1455 | 0.2895 | 0.6229 | 0.7505 |
| advanced_features | 0.2415 | 0.2868 | 0.1567 | 0.2781 | 0.6457 | 0.7534 |

## 2026 holdout results

| model | mae | rmse | r2 | winner_accuracy | winner_in_top_3_rate | ndcg_at_3 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.2465 | 0.2913 | 0.1371 | 0.2254 | 0.5434 | 0.7439 |
| history_features | 0.2392 | 0.2864 | 0.1662 | 0.2312 | 0.5896 | 0.7572 |
| advanced_features | 0.2366 | 0.2824 | 0.1892 | 0.2543 | 0.5780 | 0.7584 |

## Takeaway

- On both splits, feature engineering helps over baseline.
- On the `70/15/15` split, advanced features were strongest for overall regression quality and top-3 rate.
- On the `2026 holdout` split, advanced features are strongest for `MAE`, `RMSE`, `R2`, `winner_accuracy`, and `ndcg_at_3`.
- On the `2026 holdout` split, history features keep a slightly better `winner_in_top_3_rate`.
