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


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


SEED = 42
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data_preprocessing" / "merged_data_kr_Nan.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_CUTOFF_DATE = pd.Timestamp("2026-01-01")

RANK_COL = "순위"
SCORE_COL = "순위점수"
RACE_DATE_COL = "경주일자"
RACE_NO_COL = "경주번호"
ENTRY_NO_COL = "출전번호"
RACE_RECORD_COL = "경주기록"
BIRTH_COL = "출생일"
WEIGHT_COL = "마체중"

LEAKAGE_COLS = [RACE_RECORD_COL, RANK_COL, SCORE_COL]
DATE_COLS = [RACE_DATE_COL, BIRTH_COL]
RACE_GROUP_COL = "race_id"
HISTORY_ENTITY_COLS = ["마명", "기수번호", "조교사번호"]


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").copy()
    df[RACE_DATE_COL] = pd.to_datetime(df[RACE_DATE_COL])
    df[BIRTH_COL] = pd.to_datetime(df[BIRTH_COL])
    df[RACE_GROUP_COL] = (
        df[RACE_DATE_COL].dt.strftime("%Y-%m-%d") + "_" + df[RACE_NO_COL].astype(str)
    )
    df = df.sort_values([RACE_DATE_COL, RACE_NO_COL, ENTRY_NO_COL]).reset_index(drop=True)
    return df


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_days"] = (out[RACE_DATE_COL] - out[BIRTH_COL]).dt.days
    out["age_years"] = out["age_days"] / 365.25
    out["race_year"] = out[RACE_DATE_COL].dt.year
    out["race_month"] = out[RACE_DATE_COL].dt.month
    out["race_weekday"] = out[RACE_DATE_COL].dt.dayofweek
    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    filled_weight = out[WEIGHT_COL].fillna(out[WEIGHT_COL].median())

    for entity_col in HISTORY_ENTITY_COLS:
        group = out.groupby(entity_col, dropna=False)
        prev_count = group.cumcount()
        prefix = _history_prefix(entity_col)

        prev_score_sum = group[SCORE_COL].cumsum() - out[SCORE_COL]
        prev_rank_sum = group[RANK_COL].cumsum() - out[RANK_COL]
        prev_weight_sum = filled_weight.groupby(out[entity_col], dropna=False).cumsum() - filled_weight

        out[f"{prefix}_prev_count"] = prev_count
        out[f"{prefix}_prev_score_mean"] = np.where(prev_count > 0, prev_score_sum / prev_count, np.nan)
        out[f"{prefix}_prev_rank_mean"] = np.where(prev_count > 0, prev_rank_sum / prev_count, np.nan)
        out[f"{prefix}_prev_weight_mean"] = np.where(prev_count > 0, prev_weight_sum / prev_count, np.nan)

        prev_win_count = group[RANK_COL].transform(lambda s: s.eq(1).cumsum()) - out[RANK_COL].eq(1).astype(int)
        prev_top3_count = group[RANK_COL].transform(lambda s: s.le(3).cumsum()) - out[RANK_COL].le(3).astype(int)
        out[f"{prefix}_prev_win_count"] = prev_win_count
        out[f"{prefix}_prev_top3_count"] = prev_top3_count
        out[f"{prefix}_prev_win_rate"] = np.where(prev_count > 0, prev_win_count / prev_count, np.nan)
        out[f"{prefix}_prev_top3_rate"] = np.where(prev_count > 0, prev_top3_count / prev_count, np.nan)

        out[f"{prefix}_last_score"] = group[SCORE_COL].shift(1)
        out[f"{prefix}_last_rank"] = group[RANK_COL].shift(1)

    return out


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=DATE_COLS).copy()
    return out


def _history_prefix(entity_col: str) -> str:
    mapping = {
        "마명": "horse",
        "기수번호": "jockey",
        "조교사번호": "trainer",
    }
    return mapping[entity_col]


def split_by_date(df: pd.DataFrame) -> SplitData:
    train = df[df[RACE_DATE_COL] < TRAIN_CUTOFF_DATE].copy()
    eval_df = df[df[RACE_DATE_COL] >= TRAIN_CUTOFF_DATE].copy()

    eval_dates = np.array(sorted(eval_df[RACE_DATE_COL].dt.normalize().unique()))
    valid_end = max(1, len(eval_dates) // 2)
    valid_dates = set(eval_dates[:valid_end])
    test_dates = set(eval_dates[valid_end:])

    valid = eval_df[eval_df[RACE_DATE_COL].dt.normalize().isin(valid_dates)].copy()
    test = eval_df[eval_df[RACE_DATE_COL].dt.normalize().isin(test_dates)].copy()
    return SplitData(train=train, valid=valid, test=test)


def make_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[SCORE_COL].copy()
    x = df.drop(columns=LEAKAGE_COLS)
    return x, y


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
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def build_ridge_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=3.0, random_state=SEED)),
        ]
    )


def build_lgbm_pipeline(preprocessor: ColumnTransformer, params: dict[str, Any]) -> Pipeline:
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
    return float(sum(value / np.log2(idx + 1) for idx, value in enumerate(values, start=1)))


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

        winner_hits += int(ordered_pred.index[0] == ordered_actual.index[0])
        topk_hits += int(ordered_actual.index[0] in ordered_pred.index[: min(top_k, len(ordered_pred))])

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


def save_predictions(meta: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    pred_df = meta.copy()
    pred_df["actual_score"] = y_true.to_numpy()
    pred_df["pred_score"] = y_pred
    pred_df["pred_rank"] = (
        pred_df.groupby(RACE_GROUP_COL)["pred_score"].rank(method="first", ascending=False).astype(int)
    )
    pred_df["actual_rank_from_score"] = (
        pred_df.groupby(RACE_GROUP_COL)["actual_score"].rank(method="first", ascending=False).astype(int)
    )
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = model.named_steps["preprocess"]
    regressor: LGBMRegressor = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance = regressor.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return importance_df


def run_training(save_artifacts: bool = True) -> dict[str, Any]:
    raw_df = load_data(DATA_PATH)
    enriched_df = add_history_features(add_base_features(raw_df))
    split = split_by_date(enriched_df)

    train_df = finalize_features(split.train)
    valid_df = finalize_features(split.valid)
    test_df = finalize_features(split.test)

    train_x, train_y = make_feature_target(train_df)
    valid_x, valid_y = make_feature_target(valid_df)
    test_x, test_y = make_feature_target(test_df)

    preprocessor = build_preprocessor(train_x)

    ridge_model = build_ridge_pipeline(preprocessor)
    ridge_model.fit(train_x, train_y)
    ridge_valid_metrics = evaluate_model(
        name="ridge_baseline_history",
        model=ridge_model,
        meta=valid_df,
        x=valid_x,
        y=valid_y,
    )

    _, best_params, lgbm_valid_metrics = tune_lightgbm(
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
        name="lightgbm_history_final",
        model=final_model,
        meta=test_df,
        x=test_x,
        y=test_y,
    )
    test_pred = final_model.predict(test_x)
    feature_importance_df = extract_feature_importance(final_model)

    payload = {
        "data_path": str(DATA_PATH),
        "rows": int(raw_df.shape[0]),
        "columns": int(raw_df.shape[1]),
        "history_feature_count": int(
            len([col for col in enriched_df.columns if col.startswith(("horse_", "jockey_", "trainer_"))])
        ),
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
            "date_features_added": ["age_days", "age_years", "race_year", "race_month", "race_weekday"],
            "history_entities": HISTORY_ENTITY_COLS,
            "history_examples": [
                "horse_prev_score_mean",
                "horse_prev_win_rate",
                "jockey_last_rank",
                "trainer_prev_top3_rate",
            ],
        },
        "validation_metrics": {
            "ridge_baseline_history": ridge_valid_metrics,
            "best_lightgbm_history": lgbm_valid_metrics,
        },
        "selected_lightgbm_params": best_params,
        "test_metrics": test_metrics,
        "top_10_feature_importance": feature_importance_df.head(10).to_dict(orient="records"),
    }

    if save_artifacts:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / "rank_model_history_lightgbm.joblib"
        results_path = OUTPUT_DIR / "rank_model_history_results.json"
        preds_path = OUTPUT_DIR / "rank_model_history_test_predictions.csv"
        importance_path = OUTPUT_DIR / "rank_model_history_feature_importance.csv"

        joblib.dump(final_model, model_path)
        save_predictions(test_df, test_y, test_pred, preds_path)
        feature_importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

        payload["artifacts"] = {
            "model": str(model_path),
            "predictions": str(preds_path),
            "feature_importance": str(importance_path),
            "results": str(results_path),
        }

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main() -> None:
    payload = run_training(save_artifacts=True)
    print(json.dumps(payload["test_metrics"], ensure_ascii=False, indent=2))
    if "artifacts" in payload:
        for name, path in payload["artifacts"].items():
            print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
