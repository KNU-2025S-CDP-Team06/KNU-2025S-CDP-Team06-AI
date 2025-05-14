from fastapi import APIRouter, Request
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from .utils import parse_forecast_request
from .utils import read_csv_upload_file
import requests

forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@forecast_router.post("/daily")
async def forecast(forecast_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(forecast_file)
        forecast_result = {} # 세부 예측 로직 추가

        for store_id in forecast_result:
            url = f"http://localhost:3006/forecast/{store_id}"  # 실제 서버 주소로 수정
            data = {
                "prophet_forecast": forecast_result['prophet_forecast'],
                "xgboost_forecast": forecast_result['xgboost_forecast']
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