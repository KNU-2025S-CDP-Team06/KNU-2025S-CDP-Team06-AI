import os
import numpy as np
import pandas as pd
import optuna
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

def train_xgboost(df: pd.DataFrame, save_path: str = "./models/xgb/xgb_model.pkl"):
    """
    XGBoost 모델을 학습하고, ./models/xgb/ 경로에 저장한다.
    """

    # 날짜 정렬
    df = df.sort_values("date").reset_index(drop=True)

    # weather Label encoding
    le = LabelEncoder()
    df["weather_encoded"] = le.fit_transform(df["weather"])

    # feature 및 target 설정
    feature_cols = [
        "temp", "rain", "weather_encoded",
        "lag", "weekly_lag", "dayofweek",
        "cluster_id", "is_weekend"
    ]
    X = df[feature_cols]
    y = df["y"]
    dates = df["date"]

    # Expanding window folds 정의
    fold_boundaries = [
        ("2024-11-30", "2024-12-31"),
        ("2024-12-31", "2025-01-31"),
        ("2025-01-31", "2025-02-28"),
        ("2025-02-28", "2025-03-30")
    ]
    folds = []
    for train_end_str, test_end_str in fold_boundaries:
        train_end = pd.to_datetime(train_end_str)
        test_end = pd.to_datetime(test_end_str)
        train_idx = np.where(dates <= train_end)[0]
        test_idx = np.where((dates > train_end) & (dates <= test_end))[0]
        folds.append((train_idx, test_idx))

    # 하이퍼파라미터 튜닝 
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 5),
            "random_state": 42,
            "n_jobs": -1
        }

        fold_maes = []
        for train_idx, test_idx in folds:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_maes.append(mean_absolute_error(y_test, y_pred))

        return np.mean(fold_maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=400)

    best_params = study.best_params

    # 최적 파라미터로 모델 재학습
    final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_model, f)

    print(f"XGBoost 모델 저장 완료: {save_path}")
