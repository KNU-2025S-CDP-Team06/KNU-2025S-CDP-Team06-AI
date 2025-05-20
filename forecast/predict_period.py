import pandas as pd
import pickle
import os
from prophet import Prophet
from datetime import datetime, timedelta

def predict_period(data: dict, periods: int) -> dict:
    store_id = data["store_id"]
    date_str = data["date"]
    date = pd.to_datetime(date_str)
    cluster_id = data["cluster_id"]

    # Prophet 모델 Load
    prophet_model_path = f"./models/prophet/{store_id}.pkl"
    if not os.path.exists(prophet_model_path):
        raise FileNotFoundError(f"Prophet model not found at {prophet_model_path}")
    with open(prophet_model_path, "rb") as f:
        model: Prophet = pickle.load(f)

    # n일간 예측할 날짜 생성
    future_dates = pd.date_range(start=date, periods=periods, freq="D")
    future = pd.DataFrame({"ds": future_dates})

    # cluster_id가 2인 경우, 학기 여부 반영
    if cluster_id == 2:
        semester_ranges = [
            ("2023-03-01", "2023-06-23"), ("2023-09-01", "2023-12-22"),
            ("2024-03-04", "2024-06-20"), ("2024-09-02", "2024-12-20"),
            ("2025-03-04", "2025-06-20"), ("2025-09-01", "2025-12-20")
        ]
        future["is_semester"] = future["ds"].apply(
            lambda d: int(any(pd.to_datetime(start) <= d <= pd.to_datetime(end) for start, end in semester_ranges))
        )
        future["is_vacation"] = 1 - future["is_semester"]

    # cap, floor 설정
    future["cap"] = model.history["cap"].max() if "cap" in model.history else 1_000_0000
    future["floor"] = 0

    forecast = model.predict(future)
    y_pred_sum = float(forecast["yhat"].sum())

    return {store_id: y_pred_sum}
