from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
import pandas as pd
from xgb_utils import (
    load_and_validate_csv,
    compute_yhat_and_target,
    generate_features,
    train_and_save_xgboost
)

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
async def train_xgboost(train_file: UploadFile = File(...)):
    if train_file.content_type != "text/csv":
        return JSONResponse(content={"error": "Only CSV"}, status_code=400)

    try:
        # 2. CSV 로딩
        df = load_and_validate_csv(train_file)

        # 3. Prophet 예측을 기반으로 yhat 및 y 생성
        df_with_target = compute_yhat_and_target(df)

        # 4. 입력 피처 생성
        df_features = generate_features(df_with_target)

        # 5. XGBoost 학습 및 모델 저장
        train_and_save_xgboost(df_features)

        return JSONResponse(content={"message": "XGBoost 학습 완료 및 저장"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": f"학습 중 오류 발생: {str(e)}"}, status_code=500)

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