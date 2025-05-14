from fastapi import APIRouter, Request
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from .utils import parse_forecast_request
from .utils import read_csv_upload_file
from forecast.predict_daily import predict_daily
import requests

forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@forecast_router.post("/daily")
async def forecast(forecast_file: UploadFile = File(...)):
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

        for store_id, (y_prophet, y_xgboost) in forecast_result.items():
            url = f"http://localhost:3006/forecast/{store_id}"  # 실제 서버 주소로 수정
            data = {
                    "prophet_forecast": float(y_prophet),
                    "xgboost_forecast": float(y_xgboost)
            }
            response = requests.post(url, json=data)

        return JSONResponse(content={"message": "예측 데이터 수신 완료"}, status_code=200)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)

@forecast_router.post("/weekly")
async def forecast_weekly(request: Request):
    data = await parse_forecast_request(request)
    if not data["date"]:
        return JSONResponse(content={"error": "date is required"}, status_code=400)
    return {"date": data["date"], "prophet_forecast": 0}

@forecast_router.post("/monthly")
async def forecast_monthly(request: Request):
    data = await parse_forecast_request(request)
    if not data["date"]:
        return JSONResponse(content={"error": "date is required"}, status_code=400)
    return {"date": data["date"], "prophet_forecast": 0}