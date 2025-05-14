import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜 기준으로 store_id별 lag, weekly_lag, dayofweek, is_weekend 피처 생성
    (이전 날짜가 없으면 0으로 처리)
    """
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    feature_dfs = []

    for store_id, store_df in df.groupby("store_id"):
        store_df = store_df.copy()
        store_df = store_df.set_index("date")

        # 날짜 기반 이전 매출 값 가져오기
        store_df["rev_t-1"] = store_df["revenue"].shift(freq="1D")
        store_df["rev_t-2"] = store_df["revenue"].shift(freq="2D")
        store_df["rev_t-7"] = store_df["revenue"].shift(freq="7D")
        store_df["rev_t-14"] = store_df["revenue"].shift(freq="14D")

        # lag, weekly_lag 계산 (없으면 0으로 처리)
        store_df["lag"] = ((store_df["rev_t-1"] - store_df["rev_t-2"]) / store_df["rev_t-2"]).replace([float("inf"), -float("inf")], pd.NA).fillna(0)
        store_df["weekly_lag"] = ((store_df["rev_t-7"] - store_df["rev_t-14"]) / store_df["rev_t-14"]).replace([float("inf"), -float("inf")], pd.NA).fillna(0)

        # 요일 및 주말 여부
        store_df["dayofweek"] = store_df.index.dayofweek
        store_df["is_weekend"] = store_df["dayofweek"].isin([5, 6]).astype(int)

        store_df = store_df.reset_index()

        store_df.drop(columns=["rev_t-1", "rev_t-2", "rev_t-7", "rev_t-14"], inplace=True)

        feature_dfs.append(store_df)

    full_df = pd.concat(feature_dfs, ignore_index=True)
    return full_df
