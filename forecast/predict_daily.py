import pandas as pd
import pickle
import os
from prophet import Prophet
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
def predict_daily(data: dict) -> dict:

    # Prophet 예측
    store_id = data["store_id"]
    date_str = data["date"]
    date = pd.to_datetime(date_str)
    cluster_id = data["cluster_id"]

    # Prophet 모델을 Load
    prophet_model_path = f"./models/prophet/{store_id}.pkl"
    if not os.path.exists(prophet_model_path):
        raise FileNotFoundError(f"Prophet model not found at {prophet_model_path}")
    with open(prophet_model_path, "rb") as f:
        model: Prophet = pickle.load(f)

    future = pd.DataFrame({"ds": [date]})
    if cluster_id == 2:
        semester_ranges = [
            ("2023-03-01", "2023-06-23"), ("2023-09-01", "2023-12-22"),
            ("2024-03-04", "2024-06-20"), ("2024-09-02", "2024-12-20"),
            ("2025-03-04", "2025-06-20"), ("2025-09-01", "2025-12-20")
        ]

        in_semester = any(
            pd.to_datetime(start) <= date <= pd.to_datetime(end)
            for start, end in semester_ranges
        )

        future["is_semester"] = int(in_semester)
        future["is_vacation"] = int(not in_semester)

    future["cap"] = model.history["cap"].max() if "cap" in model.history else 1_000_0000
    future["floor"] = 0
    forecast = model.predict(future)
    y_prophet = float(forecast.iloc[0]["yhat"])

    # XGBoost 예측
    # XGBoost 입력 피처 생성 
    rev_t_1, rev_t_2 = data["rev_t-1"], data["rev_t-2"]
    rev_t_7, rev_t_14 = data["rev_t-7"], data["rev_t-14"]
    lag = (rev_t_1 - rev_t_2) / rev_t_2 if rev_t_2 != 0 else 0
    weekly_lag = (rev_t_7 - rev_t_14) / rev_t_14 if rev_t_14 != 0 else 0

    dayofweek = date.dayofweek
    is_weekend = int(dayofweek in [5, 6])

    # weather encoding
    le_path = "./models/xgb/label_encoder.pkl"

    if not os.path.exists(le_path):
        raise FileNotFoundError(f"LabelEncoder not found at {le_path}")
    with open(le_path, "rb") as f:
        le: LabelEncoder = pickle.load(f)
    
    # weather 전처리
    normalized_weather = data["weather"]
    if normalized_weather in ["Haze", "Mist", "Smoke"]:
        normalized_weather = "Fog"

    weather_encoded = int(le.transform([normalized_weather])[0])
    feature_order = [
        "temp", "rain", "weather_encoded",
        "lag", "weekly_lag", "dayofweek",
        "cluster_id", "is_weekend"
    ]
    x_row = pd.DataFrame([{
        "temp": data["temp"],
        "rain": data["rain"],
        "weather_encoded": weather_encoded,
        "lag": lag,
        "weekly_lag": weekly_lag,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "cluster_id": cluster_id,
    }])[feature_order]

    # XGBoost 모델 Load 및 예측
    xgb_model_path = f"./models/xgb/xgb_model.pkl"
    if not os.path.exists(xgb_model_path):
        raise FileNotFoundError(f"XGBoost model not found at {xgb_model_path}")
    with open(xgb_model_path, "rb") as f:
        xgb_model: XGBRegressor = pickle.load(f)

    y_xgboost = xgb_model.predict(x_row)[0]

    return {store_id: [y_prophet, y_xgboost]}
