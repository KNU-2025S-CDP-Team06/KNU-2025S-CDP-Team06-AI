from fastapi import FastAPI
from routers.train import train_router
from routers.predict import predict_router

app = FastAPI(
    title="매출 예측 시스템",
    description="Prophet/XGBoost 학습 및 예측 API",
    version="1.0.0"
)

app.include_router(train_router)
app.include_router(predict_router)