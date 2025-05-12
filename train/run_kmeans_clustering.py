import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import holidays

def run_kmeans_clustering(df_long: pd.DataFrame, k: int = 5) -> dict:
    df_long['date'] = pd.to_datetime(df_long['date'])

    # drop day-off
    df_long = df_long[df_long['revenue'] > 0]

    # holiday extraction
    years = df_long['date'].dt.year.unique()
    holiday = holidays.KR(years=years)
    holiday_dates = set(holiday.keys())

    # store_id를 기준으로 pivot
    df_wide = df_long.pivot(index="date", columns="store_id", values="revenue")

    features = {}
    fallback_ids = []  # 계산이 불가능한 매장

    for store in df_wide.columns:
        s = df_wide[store].dropna()

        # 지표 추출 
        date_series = s.index.to_series()
        weekday = date_series.dt.weekday
        is_weekend = weekday >= 5
        month = date_series.dt.month
        is_holiday = date_series.isin(holiday_dates) & (weekday < 5)

        holiday_sales = s[is_holiday]
        week_sales = s[~is_weekend]
        weekend_sales = s[is_weekend]
        semester_sales = s[month.isin([3, 4, 5, 6, 9, 10, 11, 12])]
        vacation_sales = s[month.isin([1, 2, 7, 8])]

        # 구간 중 하나라도 데이터가 없으면 계산 제외
        if (
            holiday_sales.empty or
            week_sales.empty or
            weekend_sales.empty or
            semester_sales.empty or
            vacation_sales.empty or
            s.groupby(month).mean().std() == 0
        ):
            fallback_ids.append(store)
            continue

        holiday_mean = holiday_sales.mean()
        week_diff = week_sales.mean() - weekend_sales.mean()
        semester_diff = semester_sales.mean() - vacation_sales.mean()
        yearly_seasonality_strength = s.groupby(month).mean().std()

        features[store] = {
            "holiday_mean": holiday_mean,
            "week_diff": week_diff,
            "semester_diff": semester_diff,
            "yearly_seasonality_strength": yearly_seasonality_strength
        }

    # 클러스터링
    feature_df = pd.DataFrame.from_dict(features, orient="index")

    if not feature_df.empty:
        # Row-wise scaling
        row_features = ["holiday_mean", "week_diff", "semester_diff"]
        row_scaled = feature_df[row_features].sub(
            feature_df[row_features].mean(axis=1), axis=0
        ).div(
            feature_df[row_features].std(axis=1).replace(0, 1), axis=0
        )

        # Column-wise scaling
        scaler = StandardScaler()
        col_scaled = pd.DataFrame(
            scaler.fit_transform(feature_df[["yearly_seasonality_strength"]]),
            columns=["yearly_seasonality_strength"],
            index=feature_df.index
        )

        final_scaled_df = pd.concat([row_scaled, col_scaled], axis=1)

        model = KMeans(n_clusters=k, random_state=42)
        cluster_ids = model.fit_predict(final_scaled_df)

        result = dict(zip(final_scaled_df.index, cluster_ids))
    else:
        result = {}

    # 계산 불가한 매장들은 cluster_id = 0으로 지정
    for store in fallback_ids:
        result[store] = 0

    return result
