import os
import pickle
import pandas as pd
import numpy as np
import optuna
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

def run_prophet_downtown(store_df: pd.DataFrame, store_id: int, save_dir: str = "./models/prophet/"):
    store_id_str = str(store_id)

    # preprocessing
    df = store_df.copy()
    df = df[df["revenue"] != 0]
    df = df.rename(columns={"date": "ds", "revenue": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["cap"] = df["y"].max() * 1.5
    df["floor"] = 0

    # holiday 설정: 평일에 해당하는 공휴일만 포함
    years = df["ds"].dt.year.unique().tolist()
    kr_holidays = make_holidays_df(year_list=years, country='KR')
    kr_holidays["is_weekend"] = kr_holidays["ds"].dt.weekday >= 5
    office_holidays = kr_holidays[~kr_holidays["is_weekend"]].copy()
    office_holidays["holiday"] = "weekday_only_holiday"

    # hyper-parameter tuning (최근 5개월에 대한 K-Fold 평가 기반)
    def objective(trial):
        changepoint_prior_scale = trial.suggest_categorical('changepoint_prior_scale', [0.01, 0.05, 0.1, 0.5])
        seasonality_prior_scale = trial.suggest_categorical('seasonality_prior_scale', [0.1, 1.0, 5.0, 10.0])
        holidays_prior_scale = trial.suggest_categorical('holidays_prior_scale', [0.1, 1.0, 5.0, 10.0])

        mae_scores = []
        end_date = df["ds"].max().replace(day=1)
        fold_months = [end_date - pd.DateOffset(months=i) for i in range(5, 0, -1)]

        for test_start in fold_months:
            test_end = test_start + pd.DateOffset(months=1) - timedelta(days=1)
            train = df[df["ds"] < test_start]
            valid = df[(df["ds"] >= test_start) & (df["ds"] <= test_end)]

            if len(valid) == 0:
                continue

            model = Prophet(
                growth='logistic',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                holidays=office_holidays
            )

            try:
                model.fit(train)

                future = valid[["ds"]].copy()
                future["cap"] = train["cap"].iloc[0]
                future["floor"] = 0

                forecast = model.predict(future)
                mae = mean_absolute_error(valid["y"].values, forecast["yhat"].values)
                mae_scores.append(mae)

            except Exception as e:
                print(f"[{store_id_str}] 오류 발생: {e}")
                return np.inf

        return np.mean(mae_scores) if mae_scores else np.inf

    # Optuna 튜닝
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params

    # model train
    final_model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"],
        holidays_prior_scale=best_params["holidays_prior_scale"],
        holidays=office_holidays
    )
    final_model.fit(df)

    # save model
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{store_id_str}.pkl"), "wb") as f:
        pickle.dump(final_model, f)

