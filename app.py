from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import pandas as pd
from train.xgb_utils.compute_yhat_and_target import compute_yhat_and_target
from train.xgb_utils.generate_features import generate_features
from train.xgb_utils.train_xgboost import train_xgboost
from train.prophet_utils.run_prophet_downtown import run_prophet_downtown
from train.prophet_utils.run_prophet_house import run_prophet_house
from train.prophet_utils.run_prophet_office import run_prophet_office
from train.prophet_utils.run_prophet_station import run_prophet_station
from train.prophet_utils.run_prophet_univ import run_prophet_univ

app = FastAPI(
    title="매출 예측 시스템",
    description="Prophet/XGBoost 학습 및 예측 API",
    version="1.0.0"
)

@app.post("/train/cluster")
async def train_clustering(train_file: UploadFile = File(...)):
    if train_file.content_type != "text/csv":
        return JSONResponse(content={"error": "Only CSV"}, status_code=400)
    
    df = pd.read_csv(train_file.file)
    return JSONResponse(content={"message": "Prophet 학습 데이터 수신 완료"}, status_code=200)

@app.post("/train/prophet")
async def train_prophet(train_file: UploadFile = File(...)):
    if train_file.content_type != "text/csv":
        return JSONResponse(content={"error": "Only CSV"}, status_code=400)
    
    df = pd.read_csv(train_file.file)

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
async def train_xgboost_endpoint(train_file: UploadFile = File(...)):
    if train_file.content_type != "text/csv":
        return JSONResponse(content={"error": "Only CSV file allowed"}, status_code=400)

    try:
        # CSV 로딩
        df = pd.read_csv(train_file.file)
        df["date"] = pd.to_datetime(df["date"])

        # Prophet 예측 기반 yhat 및 오차율 y 생성
        df_with_target = compute_yhat_and_target(df)

        # 입력 피처 생성
        df_features = generate_features(df_with_target)

        # XGBoost 학습 및 모델 저장
        train_xgboost(df_features)

        return JSONResponse(content={"message": "XGBoost 학습 및 저장 완료"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": f"XGBoost 학습 중 오류 발생: {str(e)}"}, status_code=500)
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