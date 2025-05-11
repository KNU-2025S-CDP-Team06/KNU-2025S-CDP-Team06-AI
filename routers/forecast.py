from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .utils import parse_forecast_request

predict_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@predict_router.post("")
async def forecast(request: Request):
    data = await parse_forecast_request(request)
    return {"prophet_forecast": 0, "xgboost_forecast": 0}

@predict_router.post("/weekly")
async def forecast_weekly(request: Request):
    data = await parse_forecast_request(request)
    if not data["date"]:
        return JSONResponse(content={"error": "date is required"}, status_code=400)
    return {"date": data["date"], "prophet_forecast": 0}

@predict_router.post("/monthly")
async def forecast_monthly(request: Request):
    data = await parse_forecast_request(request)
    if not data["date"]:
        return JSONResponse(content={"error": "date is required"}, status_code=400)
    return {"date": data["date"], "prophet_forecast": 0}