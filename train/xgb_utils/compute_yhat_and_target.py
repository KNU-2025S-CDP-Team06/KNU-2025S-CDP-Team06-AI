import pandas as pd
import numpy as np
import os
import pickle
from prophet import Prophet
from pandas.tseries.offsets import MonthEnd
# 학기 기간 정의
semester_ranges = [
    ("2023-03-01", "2023-06-23"), ("2023-09-01", "2023-12-22"),
    ("2024-03-04", "2024-06-20"), ("2024-09-02", "2024-12-20"),
    ("2025-03-04", "2025-06-20"), ("2025-09-01", "2025-12-20")
]

def is_in_semester(date):
    for start, end in semester_ranges:
        if pd.to_datetime(start) <= date <= pd.to_datetime(end):
            return True
    return False

def compute_yhat_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 store_id에 대해 Prophet 모델을 불러와 yhat 예측을 수행하고,
    revenue와 yhat을 이용하여 오차율 y = (revenue - yhat) / yhat 계산
    """
    df["date"] = pd.to_datetime(df["date"])  # datetime 변환
    store_dfs = []

    for store_id, store_df in df.groupby("store_id"):
        store_df = store_df.sort_values("date").copy()
        store_cluster_id = store_df["cluster_id"].iloc[0]

        # Prophet 모델 경로 확인
        model_path = f"./models/prophet/{store_id}.pkl"
        if not os.path.exists(model_path):
            print(f"[{store_id}] Prophet test 모델이 존재하지 않음: {model_path}")
            continue

        # Prophet 모델 로딩
        with open(model_path, "rb") as f:
            base_model: Prophet = pickle.load(f)

        # 최근 1년 예측 대상 범위
        latest_date = store_df["date"].max()
        cutoff_date = latest_date - pd.DateOffset(months=12)
        store_df = store_df[store_df["revenue"] != 0].copy()
        store_df = store_df[store_df["date"] >= cutoff_date - pd.DateOffset(months=12)]  # 2년치 확보
        store_df["ds"] = store_df["date"]

        results = []
        months = sorted(store_df["ds"].dt.to_period("M").unique())
        for i in range(12):
            if i + 12 >= len(months):
                break
            train_end = months[i + 11].to_timestamp(how="end")
            test_month = months[i + 12]
            test_start = test_month.to_timestamp()
            test_end = test_month.to_timestamp() + MonthEnd(1)

            train_df = store_df[store_df["ds"] <= train_end].copy()
            test_df = store_df[(store_df["ds"] >= test_start) & (store_df["ds"] <= test_end)].copy()
            if len(test_df) == 0:
                continue

            # cap/floor
            y_max = train_df["revenue"].max()
            y_min = train_df["revenue"].min()
            train_df["y"] = train_df["revenue"]
            train_df["cap"] = y_max * 1.1
            train_df["floor"] = y_min * 0.9 if y_min > 0 else 0

            test_df["cap"] = train_df["cap"].iloc[0]
            test_df["floor"] = train_df["floor"].iloc[0]

            train_df["ds"] = train_df["date"]
            test_df["ds"] = test_df["date"]

            # 조건부 seasonality
            if store_cluster_id == 2:
                for d in [train_df, test_df]:
                    d["is_semester"] = d["ds"].apply(lambda d: 1 if is_in_semester(d) else 0)
                    d["is_vacation"] = 1 - d["is_semester"]

            # 모델 생성 및 학습
            model = Prophet(
                growth=base_model.growth,
                yearly_seasonality=base_model.yearly_seasonality,
                weekly_seasonality=base_model.weekly_seasonality,
                daily_seasonality=base_model.daily_seasonality,
                seasonality_mode=base_model.seasonality_mode,
                changepoint_prior_scale=base_model.changepoint_prior_scale,
                seasonality_prior_scale=base_model.seasonality_prior_scale,
                holidays_prior_scale=base_model.holidays_prior_scale,
                holidays=base_model.holidays
            )
            if store_cluster_id == 2:
                model.add_seasonality("semester_weekly", period=7, fourier_order=3, condition_name="is_semester")
                model.add_seasonality("vacation_weekly", period=7, fourier_order=3, condition_name="is_vacation")

            try:
                model.fit(train_df)
                input_cols = ["ds", "cap", "floor"]
                if "is_semester" in test_df.columns:
                    input_cols += ["is_semester", "is_vacation"]

                forecast = model.predict(test_df[input_cols])
                forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
                merged = pd.merge(test_df, forecast, on="date", how="left")
                merged["y"] = (merged["revenue"] - merged["yhat"]) / merged["yhat"]
                merged["store_id"] = store_id
                results.append(merged)
            except Exception as e:
                print(f"[{store_id}] 예측 실패: {e}")
                continue

        if results:
            store_dfs.append(pd.concat(results, ignore_index=True))

    final_df = pd.concat(store_dfs, ignore_index=True)
    final_df = final_df.dropna(subset=["yhat", "y"]).reset_index(drop=True)
    return final_df