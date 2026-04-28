from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SEED = 42
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data_preprocessing" / "merged_data_kr_Nan.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_CUTOFF_DATE = pd.Timestamp("2026-01-01")

RANK_COL = "순위"
SCORE_COL = "순위점수"
LEAKAGE_COLS = ["경주기록", RANK_COL, SCORE_COL]
DATE_COLS = ["경주일자", "출생일"]
RACE_GROUP_COL = "race_id"

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").copy()
    df["경주일자"] = pd.to_datetime(df["경주일자"])
    df["출생일"] = pd.to_datetime(df["출생일"])
    df[RACE_GROUP_COL] = (
        df["경주일자"].dt.strftime("%Y-%m-%d") + "_" + df["경주번호"].astype(str)
    )
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["말나이_일"] = (out["경주일자"] - out["출생일"]).dt.days
    out["경주연도"] = out["경주일자"].dt.year
    out["경주월"] = out["경주일자"].dt.month
    out["경주요일"] = out["경주일자"].dt.dayofweek
    out["말나이_년"] = out["말나이_일"] / 365.25

    out = out.drop(columns=DATE_COLS)
    return out


def split_by_date(df: pd.DataFrame) -> SplitData:
    train = df[df["경주일자"] < TRAIN_CUTOFF_DATE].copy()
    eval_df = df[df["경주일자"] >= TRAIN_CUTOFF_DATE].copy()

    eval_dates = np.array(sorted(eval_df["경주일자"].dt.normalize().unique()))
    valid_end = max(1, len(eval_dates) // 2)
    valid_dates = set(eval_dates[:valid_end])
    test_dates = set(eval_dates[valid_end:])

    valid = eval_df[eval_df["경주일자"].dt.normalize().isin(valid_dates)].copy()
    test = eval_df[eval_df["경주일자"].dt.normalize().isin(test_dates)].copy()
    return SplitData(train=train, valid=valid, test=test)


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = feature_frame.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = feature_frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=5,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def make_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[SCORE_COL].copy()
    x = df.drop(columns=LEAKAGE_COLS)
    return x, y


def build_ridge_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=3.0, random_state=SEED)),
        ]
    )


def build_lgbm_pipeline(
    preprocessor: ColumnTransformer,
    params: dict[str, Any],
) -> Pipeline:
    model = LGBMRegressor(
        random_state=SEED,
        objective="regression",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        verbosity=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _dcg(values: list[float]) -> float:
    total = 0.0
    for idx, value in enumerate(values, start=1):
        total += value / np.log2(idx + 1)
    return total


def race_level_metrics(
    meta: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    top_k: int = 3,
) -> dict[str, float]:
    scored = meta[[RACE_GROUP_COL]].copy()
    scored["actual_score"] = y_true.to_numpy()
    scored["pred_score"] = y_pred

    winner_hits = 0
    topk_hits = 0
    ndcgs: list[float] = []

    for _, race in scored.groupby(RACE_GROUP_COL):
        ordered_pred = race.sort_values("pred_score", ascending=False)
        ordered_actual = race.sort_values("actual_score", ascending=False)

        winner_hits += int(
            ordered_pred.index[0] == ordered_actual.index[0]
        )
        topk_hits += int(
            ordered_actual.index[0] in ordered_pred.index[: min(top_k, len(ordered_pred))]
        )

        actual_relevance_by_pred = ordered_pred["actual_score"].tolist()[:top_k]
        ideal_relevance = ordered_actual["actual_score"].tolist()[:top_k]
        ideal_dcg = _dcg(ideal_relevance)
        ndcgs.append(_dcg(actual_relevance_by_pred) / ideal_dcg if ideal_dcg else 0.0)

    race_count = scored[RACE_GROUP_COL].nunique()
    return {
        "winner_accuracy": winner_hits / race_count,
        f"winner_in_top_{top_k}_rate": topk_hits / race_count,
        f"ndcg_at_{top_k}": float(np.mean(ndcgs)),
    }


def evaluate_model(
    name: str,
    model: Pipeline,
    meta: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    pred = model.predict(x)
    metrics = {
        "model": name,
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse(y, pred),
        "r2": float(r2_score(y, pred)),
    }
    metrics.update(race_level_metrics(meta=meta, y_true=y, y_pred=pred))
    return metrics


def tune_lightgbm(
    preprocessor: ColumnTransformer,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    valid_meta: pd.DataFrame,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    candidates = [
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 15,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.3,
        },
        {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 25,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 0.3,
        },
        {
            "n_estimators": 450,
            "learning_rate": 0.05,
            "num_leaves": 47,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
        },
    ]

    best_model: Pipeline | None = None
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None

    for idx, params in enumerate(candidates, start=1):
        model = build_lgbm_pipeline(preprocessor=preprocessor, params=params)
        model.fit(train_x, train_y)
        metrics = evaluate_model(
            name=f"lightgbm_candidate_{idx}",
            model=model,
            meta=valid_meta,
            x=valid_x,
            y=valid_y,
        )
        metrics["params"] = params

        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_model = model
            best_params = params
            best_metrics = metrics

    assert best_model is not None
    assert best_params is not None
    assert best_metrics is not None
    return best_model, best_params, best_metrics


def save_predictions(
    meta: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    pred_df = meta.copy()
    pred_df["actual_score"] = y_true.to_numpy()
    pred_df["pred_score"] = y_pred
    pred_df["pred_rank"] = (
        pred_df.groupby(RACE_GROUP_COL)["pred_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    pred_df["actual_rank_from_score"] = (
        pred_df.groupby(RACE_GROUP_COL)["actual_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_data(DATA_PATH)
    split = split_by_date(raw_df)

    train_df = add_features(split.train)
    valid_df = add_features(split.valid)
    test_df = add_features(split.test)

    train_x, train_y = make_feature_target(train_df)
    valid_x, valid_y = make_feature_target(valid_df)
    test_x, test_y = make_feature_target(test_df)

    preprocessor = build_preprocessor(train_x)

    ridge_model = build_ridge_pipeline(preprocessor)
    ridge_model.fit(train_x, train_y)
    ridge_valid_metrics = evaluate_model(
        name="ridge_baseline",
        model=ridge_model,
        meta=valid_df,
        x=valid_x,
        y=valid_y,
    )

    best_lgbm_model, best_params, lgbm_valid_metrics = tune_lightgbm(
        preprocessor=preprocessor,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        valid_meta=valid_df,
    )

    combined_train = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    combined_x, combined_y = make_feature_target(combined_train)

    final_preprocessor = build_preprocessor(combined_x)
    final_model = build_lgbm_pipeline(final_preprocessor, best_params)
    final_model.fit(combined_x, combined_y)

    test_metrics = evaluate_model(
        name="lightgbm_final",
        model=final_model,
        meta=test_df,
        x=test_x,
        y=test_y,
    )
    test_pred = final_model.predict(test_x)

    model_path = OUTPUT_DIR / "rank_model_lightgbm.joblib"
    results_path = OUTPUT_DIR / "rank_model_results.json"
    preds_path = OUTPUT_DIR / "rank_model_test_predictions.csv"

    joblib.dump(final_model, model_path)
    save_predictions(test_df, test_y, test_pred, preds_path)

    payload = {
        "data_path": str(DATA_PATH),
        "rows": int(raw_df.shape[0]),
        "columns": int(raw_df.shape[1]),
        "split_rows": {
            "train": int(split.train.shape[0]),
            "valid": int(split.valid.shape[0]),
            "test": int(split.test.shape[0]),
        },
        "split_rule": {
            "train_before": str(TRAIN_CUTOFF_DATE.date()),
            "validation_period": "2026 first half of race dates",
            "test_period": "2026 second half of race dates",
        },
        "feature_notes": {
            "target": SCORE_COL,
            "rank_reference": RANK_COL,
            "dropped_for_leakage": LEAKAGE_COLS,
            "date_features_added": ["말나이_일", "말나이_년", "경주연도", "경주월", "경주요일"],
        },
        "pdf_driven_choice": {
            "lecture_1": "회귀 문제로 보고 MAE/RMSE/R2를 기본 평가로 사용",
            "lecture_2": "분류 지표는 직접 타깃과 맞지 않아 보조 해석만 적용",
            "lecture_3": "앙상블/부스팅과 검증 분할을 활용해 LightGBM을 주력 모델로 선택",
            "lecture_4": "비지도/차원축소는 변수 수가 작고 해석성이 중요해 이번 버전에서는 제외",
        },
        "validation_metrics": {
            "ridge_baseline": ridge_valid_metrics,
            "best_lightgbm": lgbm_valid_metrics,
        },
        "selected_lightgbm_params": best_params,
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str(model_path),
            "predictions": str(preds_path),
        },
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload["test_metrics"], ensure_ascii=False, indent=2))
    print(f"Saved: {model_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
