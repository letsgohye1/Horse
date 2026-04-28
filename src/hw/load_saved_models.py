from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent

history_model_path = BASE_DIR / "rank_model_history_lightgbm.joblib"
advanced_model_path = BASE_DIR / "rank_model_advanced_lightgbm.joblib"

history_model = joblib.load(history_model_path)
advanced_model = joblib.load(advanced_model_path)

print(f"Loaded history model: {history_model_path}")
print(f"Loaded advanced model: {advanced_model_path}")
