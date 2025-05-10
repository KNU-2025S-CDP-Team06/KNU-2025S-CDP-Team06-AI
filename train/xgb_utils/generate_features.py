def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    store_id별로 정렬된 시계열 데이터에 대해 lag, weekly_lag, dayofweek, is_weekend 피처를 생성
    """
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    feature_dfs = []

    for store_id, store_df in df.groupby("store_id"):
        store_df = store_df.copy()

        # lag, weekly_lag 생성
        store_df["lag"] = ((store_df["revenue"].shift(1) - store_df["revenue"].shift(2)) / store_df["revenue"].shift(2)).fillna(0)
        store_df["weekly_lag"] = ((store_df["revenue"].shift(7) - store_df["revenue"].shift(14)) / store_df["revenue"].shift(14)).fillna(0)

        # 해당 날짜의 요일 및 주말 여부
        store_df["dayofweek"] = store_df["date"].dt.dayofweek
        store_df["is_weekend"] = store_df["dayofweek"].isin([5, 6]).astype(int)

        feature_dfs.append(store_df)

    full_df = pd.concat(feature_dfs, ignore_index=True)
    return full_df
