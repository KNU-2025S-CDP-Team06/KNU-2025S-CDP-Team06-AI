import os
import pickle
import pandas as pd
import numpy as np
import optuna
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

# 학기 기간 설정 
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

def is_weekend(date):
    return date.weekday() >= 5

def is_vacation(date):
    return 0 if is_in_semester(date) else 1

def run_prophet_univ(store_df: pd.DataFrame, store_id: int, save_dir: str = "./models/prophet/"):
    store_id_str = str(store_id)

    # preprocessing
    df = store_df.copy()
    df = df[df["revenue"] != 0]
    df = df.rename(columns={"date": "ds", "revenue": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    #set cap & floor
    df["cap"] = df["y"].max() * 1.1  
    df["floor"] = df["y"].min() * 0.9 if df["y"].min() > 0 else 0
    df["is_vacation"] = df["ds"].apply(is_vacation)
    df["is_semester"] = df["is_vacation"].apply(lambda x: 0 if x == 1 else 1)

    #holiday 설정: 대학가의 경우, 평일인 공휴일만 holiday로 선정 
    years = df["ds"].dt.year.unique().tolist()
    kr_holidays = make_holidays_df(year_list=years, country='KR')
    kr_holidays['in_semester'] = kr_holidays['ds'].apply(is_in_semester)
    kr_holidays['is_weekend'] = kr_holidays['ds'].apply(is_weekend)
    semester_holidays = kr_holidays[
        (kr_holidays['in_semester']) & (~kr_holidays['is_weekend'])
    ].copy()
    semester_holidays['holiday'] = 'semester_holiday'

    #hyper-parameter tuning (최근 5개월에 대한 K-Fold 평가 기반)
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
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                holidays=semester_holidays
            )
            model.add_seasonality("semester_weekly", period=7, fourier_order=3, condition_name="is_semester")
            model.add_seasonality("vacation_weekly", period=7, fourier_order=3, condition_name="is_vacation")

            try:
                model.fit(train)

                future = valid[["ds"]].copy()
                future["cap"] = train["cap"].iloc[0]
                future["floor"] = 0
                future["is_vacation"] = future["ds"].apply(is_vacation)
                future["is_semester"] = future["is_vacation"].apply(lambda x: 0 if x == 1 else 1)

                forecast = model.predict(future)
                mae = mean_absolute_error(valid["y"].values, forecast["yhat"].values)
                mae_scores.append(mae)

            except Exception as e:
                print(f"[{store_id_str}] 오류 발생: {e}")
                return np.inf

        if len(mae_scores) == 0:
            return np.inf

        return np.mean(mae_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params

    #model train
    final_model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"],
        holidays_prior_scale=best_params["holidays_prior_scale"],
        holidays=semester_holidays
    )
    final_model.add_seasonality("semester_weekly", period=7, fourier_order=3, condition_name="is_semester")
    final_model.add_seasonality("vacation_weekly", period=7, fourier_order=3, condition_name="is_vacation")
    final_model.fit(df)

    #model을 file로 변환 
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{store_id_str}.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    
    # 최근 1년 제외한 데이터로 학습한 평가용 모델 저장 (_test.pkl)
    cutoff_date = df["ds"].max() - pd.DateOffset(months=12)
    test_df = df[df["ds"] < cutoff_date].copy()
    if len(test_df) >= 180:  # 최소 데이터 확보 조건 (6개월 이상)
        test_model = Prophet(
            growth='logistic',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=best_params["changepoint_prior_scale"],
            seasonality_prior_scale=best_params["seasonality_prior_scale"],
            holidays_prior_scale=best_params["holidays_prior_scale"],
            holidays=semester_holidays
        )
        test_model.add_seasonality("semester_weekly", period=7, fourier_order=3, condition_name="is_semester")
        test_model.add_seasonality("vacation_weekly", period=7, fourier_order=3, condition_name="is_vacation")
        test_model.fit(test_df)
        with open(os.path.join(save_dir, f"{store_id_str}_test.pkl"), "wb") as f:
            pickle.dump(test_model, f)

