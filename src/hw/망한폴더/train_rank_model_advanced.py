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
RACE_GROUP_COL = "race_id"
TRAIN_CUTOFF_DATE = pd.Timestamp("2026-01-01")


@dataclass
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = df.columns.tolist()
    return {
        "split_flag": cols[0],
        "horse_name": cols[1],
        "horse_no": cols[2],
        "jockey_id": cols[3],
        "trainer_id": cols[4],
        "weight_type": cols[5],
        "entry_no": cols[6],
        "race_date": cols[7],
        "distance": cols[8],
        "race_grade": cols[9],
        "horse_type": cols[10],
        "race_no": cols[11],
        "night_flag": cols[12],
        "rank": cols[13],
        "horse_grade": cols[14],
        "race_time": cols[15],
        "field_size": cols[16],
        "track_state": cols[17],
        "weather": cols[18],
        "horse_weight": cols[19],
        "birth_date": cols[20],
        "sex": cols[21],
        "owner": cols[22],
        "country": cols[23],
        "sire": cols[24],
        "location": cols[25],
        "score": cols[26],
    }


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(path, encoding="utf-8-sig").copy()
    c = _resolve_columns(df)
    df[c["race_date"]] = pd.to_datetime(df[c["race_date"]])
    df[c["birth_date"]] = pd.to_datetime(df[c["birth_date"]])
    df[RACE_GROUP_COL] = (
        df[c["race_date"]].dt.strftime("%Y-%m-%d") + "_" + df[c["race_no"]].astype(str)
    )
    df = df.sort_values([c["race_date"], c["race_no"], c["entry_no"]]).reset_index(drop=True)
    return df, c


def add_base_features(df: pd.DataFrame, c: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["age_days"] = (out[c["race_date"]] - out[c["birth_date"]]).dt.days
    out["age_years"] = out["age_days"] / 365.25
    out["race_year"] = out[c["race_date"]].dt.year
    out["race_month"] = out[c["race_date"]].dt.month
    out["race_weekday"] = out[c["race_date"]].dt.dayofweek
    return out


def _add_entity_history_features(
    out: pd.DataFrame,
    entity_col: str,
    prefix: str,
    rank_col: str,
    score_col: str,
    weight_col: str,
) -> None:
    group = out.groupby(entity_col, dropna=False)
    prev_count = group.cumcount()
    filled_weight = out[weight_col].fillna(out[weight_col].median())
    prev_score_sum = group[score_col].cumsum() - out[score_col]
    prev_rank_sum = group[rank_col].cumsum() - out[rank_col]
    prev_weight_sum = filled_weight.groupby(out[entity_col], dropna=False).cumsum() - filled_weight

    out[f"{prefix}_prev_count"] = prev_count
    out[f"{prefix}_prev_score_mean"] = np.where(prev_count > 0, prev_score_sum / prev_count, np.nan)
    out[f"{prefix}_prev_rank_mean"] = np.where(prev_count > 0, prev_rank_sum / prev_count, np.nan)
    out[f"{prefix}_prev_weight_mean"] = np.where(prev_count > 0, prev_weight_sum / prev_count, np.nan)

    prev_win_count = group[rank_col].transform(lambda s: s.eq(1).cumsum()) - out[rank_col].eq(1).astype(int)
    prev_top3_count = group[rank_col].transform(lambda s: s.le(3).cumsum()) - out[rank_col].le(3).astype(int)
    out[f"{prefix}_prev_win_count"] = prev_win_count
    out[f"{prefix}_prev_top3_count"] = prev_top3_count
    out[f"{prefix}_prev_win_rate"] = np.where(prev_count > 0, prev_win_count / prev_count, np.nan)
    out[f"{prefix}_prev_top3_rate"] = np.where(prev_count > 0, prev_top3_count / prev_count, np.nan)
    out[f"{prefix}_last_score"] = group[score_col].shift(1)
    out[f"{prefix}_last_rank"] = group[rank_col].shift(1)


def _add_recent_trend_features(out: pd.DataFrame, horse_col: str, rank_col: str, score_col: str) -> None:
    group = out.groupby(horse_col, dropna=False)
    for lag in (1, 2, 3):
        out[f"horse_score_lag_{lag}"] = group[score_col].shift(lag)
        out[f"horse_rank_lag_{lag}"] = group[rank_col].shift(lag)

    for window in (2, 3, 5):
        out[f"horse_score_mean_last_{window}"] = (
            group[score_col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        out[f"horse_rank_mean_last_{window}"] = (
            group[rank_col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    out["horse_score_trend_1_vs_3"] = out["horse_score_lag_1"] - out["horse_score_mean_last_3"]
    out["horse_rank_trend_1_vs_3"] = out["horse_rank_lag_1"] - out["horse_rank_mean_last_3"]
    out["horse_score_trend_1_vs_5"] = out["horse_score_lag_1"] - out["horse_score_mean_last_5"]


def _add_context_history_features(
    out: pd.DataFrame,
    base_col: str,
    context_col: str,
    prefix: str,
    rank_col: str,
    score_col: str,
) -> None:
    pair = [base_col, context_col]
    group = out.groupby(pair, dropna=False)
    prev_count = group.cumcount()
    prev_score_sum = group[score_col].cumsum() - out[score_col]
    prev_rank_sum = group[rank_col].cumsum() - out[rank_col]
    prev_win_count = group[rank_col].transform(lambda s: s.eq(1).cumsum()) - out[rank_col].eq(1).astype(int)
    prev_top3_count = group[rank_col].transform(lambda s: s.le(3).cumsum()) - out[rank_col].le(3).astype(int)

    out[f"{prefix}_prev_count"] = prev_count
    out[f"{prefix}_prev_score_mean"] = np.where(prev_count > 0, prev_score_sum / prev_count, np.nan)
    out[f"{prefix}_prev_rank_mean"] = np.where(prev_count > 0, prev_rank_sum / prev_count, np.nan)
    out[f"{prefix}_prev_win_rate"] = np.where(prev_count > 0, prev_win_count / prev_count, np.nan)
    out[f"{prefix}_prev_top3_rate"] = np.where(prev_count > 0, prev_top3_count / prev_count, np.nan)
    out[f"{prefix}_last_score"] = group[score_col].shift(1)
    out[f"{prefix}_last_rank"] = group[rank_col].shift(1)


def add_advanced_history_features(df: pd.DataFrame, c: dict[str, str]) -> pd.DataFrame:
    out = add_base_features(df, c)

    _add_entity_history_features(
        out=out,
        entity_col=c["horse_name"],
        prefix="horse",
        rank_col=c["rank"],
        score_col=c["score"],
        weight_col=c["horse_weight"],
    )
    _add_entity_history_features(
        out=out,
        entity_col=c["jockey_id"],
        prefix="jockey",
        rank_col=c["rank"],
        score_col=c["score"],
        weight_col=c["horse_weight"],
    )
    _add_entity_history_features(
        out=out,
        entity_col=c["trainer_id"],
        prefix="trainer",
        rank_col=c["rank"],
        score_col=c["score"],
        weight_col=c["horse_weight"],
    )

    _add_recent_trend_features(
        out=out,
        horse_col=c["horse_name"],
        rank_col=c["rank"],
        score_col=c["score"],
    )

    _add_context_history_features(
        out=out,
        base_col=c["horse_name"],
        context_col=c["distance"],
        prefix="horse_distance",
        rank_col=c["rank"],
        score_col=c["score"],
    )
    _add_context_history_features(
        out=out,
        base_col=c["horse_name"],
        context_col=c["race_grade"],
        prefix="horse_grade_ctx",
        rank_col=c["rank"],
        score_col=c["score"],
    )
    _add_context_history_features(
        out=out,
        base_col=c["horse_name"],
        context_col=c["jockey_id"],
        prefix="horse_jockey_combo",
        rank_col=c["rank"],
        score_col=c["score"],
    )

    return out


def split_by_date(df: pd.DataFrame, race_date_col: str) -> SplitData:
    train = df[df[race_date_col] < TRAIN_CUTOFF_DATE].copy()
    eval_df = df[df[race_date_col] >= TRAIN_CUTOFF_DATE].copy()

    eval_dates = np.array(sorted(eval_df[race_date_col].dt.normalize().unique()))
    valid_end = max(1, len(eval_dates) // 2)
    valid_dates = set(eval_dates[:valid_end])
    test_dates = set(eval_dates[valid_end:])

    valid = eval_df[eval_df[race_date_col].dt.normalize().isin(valid_dates)].copy()
    test = eval_df[eval_df[race_date_col].dt.normalize().isin(test_dates)].copy()
    return SplitData(train=train, valid=valid, test=test)


def finalize_features(df: pd.DataFrame, c: dict[str, str]) -> pd.DataFrame:
    return df.drop(columns=[c["race_date"], c["birth_date"]]).copy()


def make_feature_target(df: pd.DataFrame, c: dict[str, str]) -> tuple[pd.DataFrame, pd.Series]:
    y = df[c["score"]].copy()
    x = df.drop(columns=[c["race_time"], c["rank"], c["score"]])
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
    return Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=3.0, random_state=SEED))])


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


def race_level_metrics(meta: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, top_k: int = 3) -> dict[str, float]:
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


def evaluate_model(name: str, model: Pipeline, meta: pd.DataFrame, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [
        {
            "n_estimators": 10,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        {
            "n_estimators": 25,
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 15,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.3,
        },
        {
            "n_estimators": 30,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 25,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.2,
            "reg_lambda": 0.3,
        },
        {
            "n_estimators": 50,
            "learning_rate": 0.05,
            "num_leaves": 47,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 0.5,
        },
    ]

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
            best_params = params
            best_metrics = metrics

    assert best_params is not None
    assert best_metrics is not None
    return best_params, best_metrics


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
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def run_training(save_artifacts: bool = True) -> dict[str, Any]:
    raw_df, c = load_data(DATA_PATH)
    enriched_df = add_advanced_history_features(raw_df, c)
    split = split_by_date(enriched_df, race_date_col=c["race_date"])

    train_df = finalize_features(split.train, c)
    valid_df = finalize_features(split.valid, c)
    test_df = finalize_features(split.test, c)

    train_x, train_y = make_feature_target(train_df, c)
    valid_x, valid_y = make_feature_target(valid_df, c)
    test_x, test_y = make_feature_target(test_df, c)

    preprocessor = build_preprocessor(train_x)

    ridge_model = build_ridge_pipeline(preprocessor)
    ridge_model.fit(train_x, train_y)
    ridge_valid_metrics = evaluate_model(
        name="ridge_baseline_advanced",
        model=ridge_model,
        meta=valid_df,
        x=valid_x,
        y=valid_y,
    )

    best_params, lgbm_valid_metrics = tune_lightgbm(
        preprocessor=preprocessor,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        valid_meta=valid_df,
    )

    combined_train = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    combined_x, combined_y = make_feature_target(combined_train, c)
    final_preprocessor = build_preprocessor(combined_x)
    final_model = build_lgbm_pipeline(final_preprocessor, best_params)
    final_model.fit(combined_x, combined_y)

    test_metrics = evaluate_model(
        name="lightgbm_advanced_final",
        model=final_model,
        meta=test_df,
        x=test_x,
        y=test_y,
    )
    test_pred = final_model.predict(test_x)
    feature_importance_df = extract_feature_importance(final_model)

    advanced_feature_prefixes = (
        "horse_score_lag_",
        "horse_rank_lag_",
        "horse_score_mean_last_",
        "horse_rank_mean_last_",
        "horse_score_trend_",
        "horse_rank_trend_",
        "horse_distance_",
        "horse_grade_ctx_",
        "horse_jockey_combo_",
    )

    payload = {
        "data_path": str(DATA_PATH),
        "rows": int(raw_df.shape[0]),
        "columns": int(raw_df.shape[1]),
        "feature_counts": {
            "history_features": int(
                len([col for col in enriched_df.columns if col.startswith(("horse_", "jockey_", "trainer_"))])
            ),
            "advanced_context_features": int(
                len([col for col in enriched_df.columns if col.startswith(advanced_feature_prefixes)])
            ),
        },
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
            "target": "normalized_rank_score",
            "dropped_for_leakage": ["race_time", "rank", "score"],
            "recent_trend_examples": [
                "horse_score_lag_1",
                "horse_score_mean_last_3",
                "horse_rank_mean_last_5",
                "horse_score_trend_1_vs_3",
            ],
            "distance_grade_examples": [
                "horse_distance_prev_score_mean",
                "horse_distance_prev_top3_rate",
                "horse_grade_ctx_prev_score_mean",
                "horse_grade_ctx_last_rank",
            ],
            "combo_examples": [
                "horse_jockey_combo_prev_count",
                "horse_jockey_combo_prev_score_mean",
                "horse_jockey_combo_prev_win_rate",
                "horse_jockey_combo_last_score",
            ],
        },
        "validation_metrics": {
            "ridge_baseline_advanced": ridge_valid_metrics,
            "best_lightgbm_advanced": lgbm_valid_metrics,
        },
        "selected_lightgbm_params": best_params,
        "test_metrics": test_metrics,
        "top_15_feature_importance": feature_importance_df.head(15).to_dict(orient="records"),
    }

    if save_artifacts:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = OUTPUT_DIR / "rank_model_advanced_lightgbm.joblib"
        results_path = OUTPUT_DIR / "rank_model_advanced_results.json"
        preds_path = OUTPUT_DIR / "rank_model_advanced_test_predictions.csv"
        importance_path = OUTPUT_DIR / "rank_model_advanced_feature_importance.csv"

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
