from fastapi import APIRouter, Request
from fastapi import APIRouter, UploadFile, File, Path
from fastapi.responses import JSONResponse
from .utils import parse_forecast_request, read_csv_upload_file, get_jwt
from forecast.predict_daily import predict_daily
from forecast.predict_period import predict_period
import requests
from config import config
from datetime import datetime

forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@forecast_router.post("/{period}")
async def forecast(
    period: str = Path(..., regex="^(daily|weekly|monthly)$"),
    forecast_file: UploadFile = File(...)
):
    try:
        df = read_csv_upload_file(forecast_file)

        df = df.rename(columns={
            "precipitation": "rain",
            "feeling": "temp"
        })

        forecast_result = {}

        if period == "daily":
            # daily 처리 로직
            pass

        elif period == "weekly":
            # daily + weekly 처리 로직
            pass

        elif period == "monthly":
            # daily + weekly+ monthly 처리 로직
            pass

        # JWT 인증
        headers = {
            "Authorization": f"Bearer {get_jwt()}"
        }
        for store_id, (y_prophet, y_xgboost, date) in forecast_result.items():
            url = f"{config.BACKEND_URL}/forecast"

            # 여기서 그냥 한 번에 합쳐서 보내면 됨
            '''
            data = {
	            "store_id": store_id,
                "prophet_forecast": int(y_prophet),
                "xgboost_forecast": float(y_xgboost),
                "dateTime" : datetime.combine(date, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%S”), // 날짜 데이터 형변환
                “weeky_forecast” : int(y_prophet), // 없으면 null 가능
                “monthly_forecast” : int(y_prophet), // 없으면 null 가능
            }
            '''
            
            # predcit_daily 함수 보면 날짜 반환할 수 있게 바꿔놨으니까 그거 참고하면 됨
            data = {
                "store_id": store_id,
                "prophet_forecast": float(y_prophet),
                "xgboost_forecast": float(y_xgboost),
                "dateTime": datetime.combine(date, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%S")
            }
            response = requests.post(url, json=data, headers=headers)

            if response.status_code != 204:
                raise ValueError(f"{store_id} 저장 실패: {response.status_code} - {response.text}")

        return JSONResponse(content={"message": f"{period} 예측 데이터 수신 완료"}, status_code=200)

    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)


'''
여기서부터는 원래 있던 코드
참고해서 위에꺼 작성하고 다 하면 밑에건 다 지우면 됨
'''
@forecast_router.post("/daily")
async def forecast_daily(forecast_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(forecast_file)
        df = df.rename(columns={
            "precipitation": "rain",
            "feeling": "temp"
        })

        forecast_result = {} # 세부 예측 로직 추가

        for _, row in df.iterrows():
            # 한 행을 dict로 변환하여 predict_daily에 전달
            input_dict = {
                "store_id": int(row["store_id"]),
                "date": row["date"],
                "temp": float(row["temp"]),
                "rain": float(row["rain"]),
                "weather": row["weather"],
                "cluster_id": int(row["cluster_id"]),
                "rev_t-1": float(row["rev_t-1"]),
                "rev_t-2": float(row["rev_t-2"]),
                "rev_t-7": float(row["rev_t-7"]),
                "rev_t-14": float(row["rev_t-14"]),
            }

            prediction = predict_daily(input_dict)
            forecast_result.update(prediction)

        # JWT 인증
        headers = {
            "Authorization": f"Bearer {get_jwt()}"
        }
        for store_id, (y_prophet, y_xgboost, date) in forecast_result.items():
            url = f"{config.BACKEND_URL}/forecast"

            data = {
                "store_id": store_id,
                "prophet_forecast": int(y_prophet),
                "xgboost_forecast": float(y_xgboost),
                "dateTime" : datetime.combine(date, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%S")
            }
            response = requests.post(url, json=data, headers=headers)

            if response.status_code != 204:
                raise ValueError(f"{store_id} 저장 실패: {response.status_code} - {response.text}")
            
        return JSONResponse(content={"message": "예측 데이터 수신 완료"}, status_code=200)
    
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)

@forecast_router.post("/weekly")
async def forecast_weekly(forecast_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(forecast_file)
        forecast_result = {} # 세부 예측 로직 추가

        for _, row in df.iterrows():
            # 한 행을 dict로 변환하여 predict_weekly에 전달
            input_dict = {
                "store_id": int(row["store_id"]),
                "date": row["date"],
                "cluster_id": int(row["cluster_id"]),
            }

            prediction = predict_period(input_dict, periods = 7) # 7일 간의 매출을 예측
            forecast_result.update(prediction)

        # JWT 인증
        headers = {
            "Authorization": f"Bearer {get_jwt()}"
        }
        for store_id, weekly_forecast in forecast_result.items():
            url = f"{config.BACKEND_URL}/forecast"
            data = {
                "store_id": store_id,
                "prophet_forecast": float(weekly_forecast)
            }
            print(data)

            response = requests.post(url, json=data, headers=headers)

            if response.status_code != 204:
                raise ValueError(f"{store_id} 저장 실패: {response.status_code} - {response.text}")

        return JSONResponse(content={"message": "예측 데이터 수신 완료"}, status_code=200)
    
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)

@forecast_router.post("/monthly")
async def forecast_monthly(forecast_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(forecast_file)
        forecast_result = {} # 세부 예측 로직 추가

        for _, row in df.iterrows():
            # 한 행을 dict로 변환하여 predict_daily에 전달
            input_dict = {
                "store_id": int(row["store_id"]),
                "date": row["date"],
                "cluster_id": int(row["cluster_id"]),
            }

            prediction = predict_period(input_dict, periods = 30) # 30일 간의 매출을 예측
            forecast_result.update(prediction)


        # JWT 인증
        headers = {
            "Authorization": f"Bearer {get_jwt()}"
        }
        for store_id, monthly_forecast in forecast_result.items():
            url = f"{config.BACKEND_URL}/forecast"
            data = {
                "store_id": store_id,
                "prophet_forecast": float(monthly_forecast)
            }

            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code != 204:
                raise ValueError(f"{store_id} 저장 실패: {response.status_code} - {response.text}")

        return JSONResponse(content={"message": "예측 데이터 수신 완료"}, status_code=200)
    
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)