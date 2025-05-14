from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from .utils import read_csv_upload_file, get_prophet_function
from train.xgb_utils.compute_yhat_and_target import compute_yhat_and_target
from train.xgb_utils.generate_features import generate_features
from train.xgb_utils.train_xgboost import train_xgboost
from train.run_kmeans_clustering import run_kmeans_clustering
import requests
from config import config

train_router = APIRouter(prefix="/train", tags=["Training"])

@train_router.post("/cluster")
async def train_clustering(train_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(train_file)
        cluster_result = run_kmeans_clustering(df)
        print(cluster_result)
        for store_id in cluster_result:
            url = f"{config.BACKEND_URL}/train/{store_id}"
            data = {
                "cluster": int(cluster_result[store_id])  # 예시 클러스터 값
            }
            response = requests.patch(url, json=data)

        return JSONResponse(content={"message": "학습 데이터 수신 완료"}, status_code=200)
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)

@train_router.post("/prophet")
async def train_prophet(train_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(train_file)
        for store_id, store_df in df.groupby("store_id"):
            cluster_id = store_df["cluster_id"].iloc[0]
            prophet_func = get_prophet_function(cluster_id)
            if prophet_func:
                prophet_func(store_df[["date", "revenue"]], store_id)
        return JSONResponse(content={"message": "Prophet 학습 완료"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@train_router.post("/xgboost")
async def train_xgboost_endpoint(train_file: UploadFile = File(...)):
    try:
        df = read_csv_upload_file(train_file)
        df["date"] = pd.to_datetime(df["date"])
        df_with_target = compute_yhat_and_target(df)
        df_features = generate_features(df_with_target)
        train_xgboost(df_features)
        return JSONResponse(content={"message": "XGBoost 학습 및 저장 완료"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)