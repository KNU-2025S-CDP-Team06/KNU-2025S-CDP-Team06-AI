from fastapi import APIRouter, Request
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from .utils import parse_forecast_request, read_csv_upload_file, get_jwt
from forecast.predict_daily import predict_daily
from forecast.predict_period import predict_period
from forecast.check_if_holiday import check_if_holiday
import requests
from config import config
from datetime import datetime
import pandas as pd
forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@forecast_router.post("/")
async def forecast_daily(forecast_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(forecast_file)

        forecast_result = [] # 세부 예측 로직 추가

        for _, row in df.iterrows():
            input_dict = {
                "store_id": int(row["store_id"]),
                "date": row["date"],
                "temp": float(row["temp"]),
                "rain": float(row["rain"]),
                "weather": row["weather"],
                "cluster_id": int(row["cluster_id"]),
                "rev_t-1": float(row["rev_t-1"]),
                "rev_t-2": float(row["rev_t-2"]),
                "rev_t-3": float(row["rev_t-3"]),
                "rev_t-4": float(row["rev_t-4"]),
                "rev_t-5": float(row["rev_t-5"]),
                "rev_t-6": float(row["rev_t-6"]),
                "rev_t-7": float(row["rev_t-7"]),
                "rev_t-8": float(row["rev_t-8"]),
                "rev_t-9": float(row["rev_t-9"]),
                "rev_t-10": float(row["rev_t-10"]),
                "rev_t-11": float(row["rev_t-11"]),
                "rev_t-12": float(row["rev_t-12"]),
                "rev_t-13": float(row["rev_t-13"]),
                "rev_t-14": float(row["rev_t-14"]),
            }

            # 1일차 예측 (Prophet + XGBoost)
            result_day1 = predict_daily(input_dict)
            y_prophet, y_xgboost, date = result_day1[input_dict["store_id"]]
            forecast_result.append({
                "store_id": input_dict["store_id"],
                "date": date,
                "prophet_forecast": float(y_prophet),
                "xgboost_forecast": float(y_xgboost)
            })

            # 2~62일차 예측 (Prophet only)
            period_result = predict_period(input_dict, periods=62)
            future_start_date = pd.to_datetime(input_dict["date"]) + pd.Timedelta(days=1)
            
            for i, yhat in enumerate(period_result[input_dict["store_id"]]):
                forecast_result.append({
                    "store_id": input_dict["store_id"],
                    "date": (future_start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"),
                    "prophet_forecast": float(yhat),
                    "xgboost_forecast": None
                })
            # 예측 후 휴일 요일 판단
            holiday_weekdays = check_if_holiday(input_dict)

            # forecast_result에서 해당 요일 prophet 예측값 0으로 수정
            for row in forecast_result:
                if row["store_id"] != input_dict["store_id"]:
                    continue
                forecast_dt = pd.to_datetime(row["date"])
                if forecast_dt.weekday() in holiday_weekdays:
                    row["prophet_forecast"] = 0.0

        
        # JWT 인증
        headers = {
            "Authorization": f"Bearer {get_jwt()}"
        } 
        
        for row in forecast_result:
            data = {
                "store_id": row["store_id"],
                "prophet_forecast": row["prophet_forecast"],
                "xgboost_forecast": row["xgboost_forecast"] * 0.1 if row["xgboost_forecast"] is not None else None,
                "date_time": datetime.strptime(str(row["date"]), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            response = requests.post(f"{config.BACKEND_URL}/forecast", json=data, headers=headers)
            if response.status_code != 204:
                raise ValueError(f"{row['store_id']} 저장 실패: {response.status_code} - {response.text}")
            
        return JSONResponse(content={"message": "예측 데이터 수신 완료"}, status_code=200)
    
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)