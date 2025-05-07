import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import holidays

def run_kmeans_clustering(df_long: pd.DataFrame, k: int = 5) -> dict:

    df_long['date'] = pd.to_datetime(df_long['date'])

    #Extracting holidays
    years = df_long['date'].dt.year.unique()
    holiday = holidays.KR(years=years)
    holiday_dates = set(holiday.keys())

    # pivot: date x store_id
    df_wide = df_long.pivot(index="date", columns="store_id", values="revenue")

    features = {}
    for store in df_wide.columns:
        s = df_wide[store].dropna()

        date_series = s.index.to_series()
        weekday = date_series.dt.weekday
        is_weekend = weekday >= 5
        month = date_series.dt.month
        is_holiday = date_series.isin(holiday_dates) & (weekday < 5)

        # find holiday_diff
        holiday_operated = s[is_holiday & (s > 0)]
        holiday_mean = holiday_operated.mean()

        # find week_diff 
        week_mean = s[~is_weekend & (s > 0)].mean()
        weekend_mean = s[is_weekend & (s > 0)].mean()
        week_diff = week_mean - weekend_mean

        # find semester_diff 
        semester_mask = month.isin([3, 4, 5, 6, 9, 10, 11, 12])
        vacation_mask = month.isin([1, 2, 7, 8])
        semester_mean = s[semester_mask & (s > 0)].mean()
        vacation_mean = s[vacation_mask & (s > 0)].mean()
        semester_diff = semester_mean - vacation_mean

        # find yearly_seasonaity_strength
        yearly_seasonality_strength = s.groupby(month).mean().std()

        features[store] = {
            "holiday_mean": holiday_mean,
            "week_diff": week_diff,
            "semester_diff": semester_diff,
            "yearly_seasonality_strength": yearly_seasonality_strength
        }

    feature_df = pd.DataFrame.from_dict(features, orient="index")

    # Row-wise scaling
    row_features = ["holiday_mean", "week_diff", "semester_diff"]
    row_scaled = feature_df[row_features].sub(
        feature_df[row_features].mean(axis=1), axis=0
    ).div(
        feature_df[row_features].std(axis=1), axis=0
    )

    # Column-wise scaling
    scaler = StandardScaler()
    col_scaled = pd.DataFrame(
        scaler.fit_transform(feature_df[["yearly_seasonality_strength"]]),
        columns=["yearly_seasonality_strength"],
        index=feature_df.index
    )

    final_scaled_df = pd.concat([row_scaled, col_scaled], axis=1)

    # K-Means clustering
    model = KMeans(n_clusters=k, random_state=42)
    cluster_ids = model.fit_predict(final_scaled_df)

    return dict(zip(final_scaled_df.index, cluster_ids))

