import os
import numpy as np
import pandas as pd
import optuna
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import sys

def train_xgboost(df: pd.DataFrame, save_path: str = "./models/xgb/xgb_model.pkl"):
    """
    XGBoost 모델을 학습하고, ./models/xgb/ 경로에 저장한다.
    """

    # 날짜 정렬
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.to_period("M")

    # 이상치 제거 전 샘플 수
    n_before_outlier_removal = len(df)
    df.to_csv("./models/xgb/before_outlier_removal.csv", index=False)
    
    # Prophet 예측 신뢰구간 기반 이상치 제거
    if "yhat_lower" in df.columns and "yhat_upper" in df.columns:
        df = df[(df["revenue"] >= df["yhat_lower"]) & (df["revenue"] <= df["yhat_upper"])].copy()
        df.reset_index(drop=True, inplace=True)

    # 이상치 제거 후 샘플 수
    n_after_outlier_removal = len(df)
    df.to_csv("./models/xgb/after_outlier_removal.csv", index=False)

    print("n_before_outlier_removal: ", n_before_outlier_removal, "n_after_outlier_removal: ", n_after_outlier_removal)
    sys.stdout.flush()
    # weather category 통합
    df["weather"] = df["weather"].replace({
        "Haze": "Fog",
        "Mist": "Fog",
        "Smoke": "Fog"
    })

    # weather Label Encoding
    le = LabelEncoder()
    df["weather_encoded"] = le.fit_transform(df["weather"])

    # Label Encoder 저장
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

    # Expanding window folds 정의 (최근 4개월을 대상으로 Test 진행)
    recent_months = sorted(df["month"].unique())[-4:]
    folds = []
    for i in range(0,len(recent_months)):
        test_month = recent_months[i]
        train_months = recent_months[:i]  # test 월 이전까지
        train_idx = df.index[df["month"].isin(train_months)]
        test_idx = df.index[df["month"] == test_month]
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
    study.optimize(objective, n_trials= 80)
    best_params = study.best_params

    # 최적 파라미터로 Fold별 성능 측정 및 모델 학습
    mae_list = []
    for i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_test) == 0 or len(y_train) == 0:
            continue

        model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)
    
    # 로그 저장 (이상치 정보 + MAE + 하이퍼파라미터)
    log_path = "./models/xgb/xgb_log.txt"
    with open(log_path, "w") as f:
        f.write("=== XGBoost 학습 로그 ===\n")
        f.write(f"이상치 제거 전 샘플 수: {n_before_outlier_removal}\n")
        f.write(f"이상치 제거 후 샘플 수: {n_after_outlier_removal}\n\n")

        for i, mae in enumerate(mae_list):
            f.write(f"Fold {i + 1}: MAE = {mae:.4f}\n")
        f.write(f"\nAverage MAE: {np.mean(mae_list):.4f}\n")
        f.write(f"\nMedian MAE: {np.median(mae_list):.4f}\n")

        f.write("\nBest Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
