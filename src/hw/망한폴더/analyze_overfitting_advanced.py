from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor

from train_rank_model_advanced import (
    DATA_PATH,
    OUTPUT_DIR,
    SEED,
    add_advanced_history_features,
    build_preprocessor,
    finalize_features,
    load_data,
    make_feature_target,
    split_by_date,
)


def run_overfitting_analysis() -> dict:
    raw_df, c = load_data(DATA_PATH)
    enriched_df = add_advanced_history_features(raw_df, c)
    split = split_by_date(enriched_df, race_date_col=c["race_date"])

    train_df = finalize_features(split.train, c)
    valid_df = finalize_features(split.valid, c)
    test_df = finalize_features(split.test, c)

    combined_train = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    train_x, train_y = make_feature_target(combined_train, c)
    test_x, test_y = make_feature_target(test_df, c)

    preprocessor = build_preprocessor(train_x)
    train_x_t = preprocessor.fit_transform(train_x)
    test_x_t = preprocessor.transform(test_x)

    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": SEED,
        "objective": "regression",
        "verbosity": -1,
    }

    model = LGBMRegressor(**params)
    model.fit(
        train_x_t,
        train_y,
        eval_set=[(train_x_t, train_y), (test_x_t, test_y)],
        eval_names=["train", "test"],
        eval_metric="rmse",
    )

    evals_result = model.evals_result_
    train_rmse = evals_result["train"]["rmse"]
    test_rmse = evals_result["test"]["rmse"]

    loss_df = pd.DataFrame(
        {
            "iteration": range(1, len(train_rmse) + 1),
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        }
    )
    loss_df["gap"] = loss_df["test_rmse"] - loss_df["train_rmse"]

    best_idx = int(loss_df["test_rmse"].idxmin())
    best_row = loss_df.iloc[best_idx]
    final_row = loss_df.iloc[-1]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "advanced_overfitting_curve.csv"
    png_path = OUTPUT_DIR / "advanced_overfitting_curve.png"
    json_path = OUTPUT_DIR / "advanced_overfitting_summary.json"

    loss_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_df["iteration"], loss_df["train_rmse"], label="Train RMSE", linewidth=2)
    plt.plot(loss_df["iteration"], loss_df["test_rmse"], label="Test RMSE", linewidth=2)
    plt.axvline(best_row["iteration"], color="gray", linestyle="--", linewidth=1.5, label="Best test iteration")
    plt.scatter([best_row["iteration"]], [best_row["test_rmse"]], color="red", zorder=5)
    plt.title("Advanced Model Overfitting Check")
    plt.xlabel("Boosting Iteration")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    summary = {
        "split_rule": {
            "train": "before 2026-01-01 plus 2026 validation half combined for final train curve",
            "test": "2026 second half race dates",
        },
        "artifacts": {
            "curve_csv": str(csv_path),
            "curve_png": str(png_path),
        },
        "best_test_iteration": int(best_row["iteration"]),
        "best_train_rmse": float(best_row["train_rmse"]),
        "best_test_rmse": float(best_row["test_rmse"]),
        "best_gap": float(best_row["gap"]),
        "final_iteration": int(final_row["iteration"]),
        "final_train_rmse": float(final_row["train_rmse"]),
        "final_test_rmse": float(final_row["test_rmse"]),
        "final_gap": float(final_row["gap"]),
        "overfitting_signal": bool(final_row["test_rmse"] > best_row["test_rmse"] and final_row["gap"] > best_row["gap"]),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    summary = run_overfitting_analysis()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
