from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import pandas as pd

app = FastAPI(
    title="매출 예측 시스템",
    description="Prophet/XGBoost 학습 및 예측 API",
    version="1.0.0"
)


@app.post("/train/prophet")
async def train_prophet(revenue_file: UploadFile = File(...)):
    if revenue_file.content_type != "text/csv":
        return JSONResponse(content={"error": "Only CSV"}, status_code=400)
    
    df = pd.read_csv(revenue_file.file)

    return JSONResponse(content={"message": "Prophet 학습 데이터 수신 완료"}, status_code=200)


@app.post("/train/xgboost")
async def train_xgboost(
    revenue_file: UploadFile = File(...),
    weather_file: UploadFile = File(...)
):
    if revenue_file.content_type != "text/csv" or weather_file.content_type != "text/csv":
        return JSONResponse(content={"error": "Only CSV"}, status_code=400)

    pd.read_csv(revenue_file.file)
    pd.read_csv(weather_file.file)

    return JSONResponse(content={"message": "XGBoost 학습 데이터 수신 완료"}, status_code=200)


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    store_id = data.get("store_id")
    date = data.get("date")
    weather = data.get("weather", {})

    return {
        "prophet_forecast": 0,
        "xgboost_forecast": 0
    }


@app.post("/predict/weekly")
async def predict_weekly(request: Request):
    data = await request.json()
    store_id = data.get("store_id")
    date = data.get("date")
    weather = data.get("weather", {})

    if not date:
        return JSONResponse(content={"error": "Bad Request: date is required"}, status_code=400)

    return {
        "date": date,
        "prophet_forecast": 0
    }


@app.post("/predict/monthly")
async def predict_monthly(request: Request):
    data = await request.json()
    store_id = data.get("store_id")
    date = data.get("date")
    weather = data.get("weather", {})

    if not date:
        return JSONResponse(content={"error": "Bad Request: date is required"}, status_code=400)

    return {
        "date": date,
        "prophet_forecast": 0
    }