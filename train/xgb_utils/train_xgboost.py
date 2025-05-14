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
    df["month"] = df["date"].dt.to_period("M")
    
    # 이상치 제거: 매출이 월 평균 대비 ±50% 이상인 경우 제거
    
    monthly_avg = df.groupby("month")["y"].transform("mean")
    lower = monthly_avg * 0.5
    upper = monthly_avg * 1.5
    df = df[(df["y"] >= lower) & (df["y"] <= upper)].copy()
    df.reset_index(drop=True, inplace=True)  

    # weather category 통합
    df["weather"] = df["weather"].replace({
        "Haze": "Fog",
        "Mist": "Fog",
        "Smoke": "Fog"
    })

    # weather Label Encoding
    le = LabelEncoder()
    df["weather_encoded"] = le.fit_transform(df["weather"])

    #Label Encoder 저장
    os.makedirs("./models/xgb", exist_ok=True)
    with open("./models/xgb/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # feature 및 target 설정
    feature_cols = [
        "temp", "rain", "weather_encoded",
        "lag", "weekly_lag", "dayofweek",
        "cluster_id", "is_weekend"
    ]
    X = df[feature_cols]
    y = df["y"]
    dates = df["date"]

    # Expanding window folds 정의 (최근 4개월을 대상으로 Test 진행)
    recent_months = sorted(df["month"].unique())[-4:]
    folds = []
    for i in range(len(recent_months)):
        train_end = recent_months[i]
        test_end = recent_months[i]
        train_idx = df.index[df["month"] <= train_end]
        test_idx = df.index[df["month"] == test_end]
        folds.append((train_idx, test_idx))
    df.drop(columns=["month"], inplace=True)

    # Optuna 튜닝
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

            if len(y_test) == 0 or len(y_train) == 0:
                continue

            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_maes.append(mean_absolute_error(y_test, y_pred))

        return np.mean(fold_maes) if fold_maes else float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_params

    # 최적 파라미터로 모델 재학습
    final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(final_model, f)