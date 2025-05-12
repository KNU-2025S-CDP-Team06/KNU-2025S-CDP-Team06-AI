import pandas as pd
from fastapi import UploadFile, Request

def read_csv_upload_file(upload_file: UploadFile):
    if upload_file.content_type != "text/csv":
        raise ValueError("Only CSV files are allowed")
    return pd.read_csv(upload_file.file)

def get_prophet_function(cluster_id: int):
    from train.prophet_utils.run_prophet_downtown import run_prophet_downtown
    from train.prophet_utils.run_prophet_house import run_prophet_house
    from train.prophet_utils.run_prophet_office import run_prophet_office
    from train.prophet_utils.run_prophet_station import run_prophet_station
    from train.prophet_utils.run_prophet_univ import run_prophet_univ

    return {
        0: run_prophet_office,
        1: run_prophet_house,
        2: run_prophet_univ,
        3: run_prophet_downtown,
        4: run_prophet_station
    }.get(cluster_id)

async def parse_forecast_request(request: Request):
    data = await request.json()
    return {
        "store_id": data.get("store_id"),
        "date": data.get("date"),
        "weather": data.get("weather", {})
    }