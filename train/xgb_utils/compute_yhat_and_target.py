import pandas as pd
import numpy as np
import os
import pickle
from prophet import Prophet

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
            print(f"[{store_id}] Prophet 모델이 존재하지 않음: {model_path}")
            continue

        # Prophet 모델 로딩
        with open(model_path, "rb") as f:
            model: Prophet = pickle.load(f)

        store_df["month"] = store_df["date"].dt.to_period("M")
        cleaned_df = store_df[store_df["revenue"] != 0].copy()
        cleaned_df.drop(columns=["month"], inplace=True)

        # Prophet 예측을 위한 컬럼 설정
        cleaned_df["ds"] = cleaned_df["date"]
        cleaned_df["cap"] = cleaned_df["revenue"].max() * 1.5
        cleaned_df["floor"] = 0

        # 대학가 상권일 경우, is_semester, is_vacation 값 계산
        if store_cluster_id == 2:
            cleaned_df["is_semester"] = cleaned_df["ds"].apply(lambda d: 1 if is_in_semester(d) else 0)
            cleaned_df["is_vacation"] = cleaned_df["is_semester"].apply(lambda x: 0 if x == 1 else 1)

        # 예측 수행
        input_cols = ["ds", "cap", "floor"]
        if "is_semester" in cleaned_df.columns:
            input_cols.append("is_semester")
        if "is_vacation" in cleaned_df.columns:
            input_cols.append("is_vacation")

        try:
            forecast = model.predict(cleaned_df[input_cols])
        except Exception as e:
            print(f"[{store_id}] 예측 실패: {e}")
            continue

        yhat_df = forecast[["ds", "yhat"]].rename(columns={"ds": "date"})
        merged = pd.merge(store_df, yhat_df, on="date", how="left") 

        # 타겟 계산
        merged["y"] = (merged["revenue"] - merged["yhat"]) / merged["yhat"]
        merged["store_id"] = store_id

        store_dfs.append(merged)

    # 모든 store 데이터 통합
    final_df = pd.concat(store_dfs, ignore_index=True)
    final_df = final_df.dropna(subset=["yhat", "y"]).reset_index(drop=True)
    return final_df
