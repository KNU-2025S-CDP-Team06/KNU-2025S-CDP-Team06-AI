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

    for store_id, store_df in df.groupby("store_id"):
        store_cluster_id = store_df["cluster_id"].iloc[0]
        store_df = store_df[["date", "revenue"]]  

        if store_cluster_id == 0:
            run_prophet_office(store_df, store_id)
        elif store_cluster_id == 1:
            run_prophet_station(store_df, store_id)
        elif store_cluster_id == 2:
            run_prophet_house(store_df, store_id)
        elif store_cluster_id == 3:
            run_prophet_downtown(store_df, store_id)
        elif store_cluster_id == 4:
            run_prophet_univ(store_df, store_id)

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