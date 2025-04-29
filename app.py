from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)


@app.route('/train/prophet', methods=['POST'])
def train_prophet():
    data = request.get_json()
    print("📌 [Prophet 학습] 받은 데이터:", data.get("revenue_data"))
    return '', 200


@app.route('/train/xgboost', methods=['POST'])
def train_xgboost():
    data = request.get_json()
    print("📌 [XGBoost 학습] 매출 데이터:", data.get("revenue_data"))
    print("📌 [XGBoost 학습] 날씨 데이터:", data.get("weather_data"))
    return '', 200


@app.route('/predict/<int:store_id>', methods=['GET'])
def predict(store_id):
    date = request.args.get('date')
    print(f"📌 [예측 요청] 매장 ID: {store_id}, 날짜: {date}")
    return jsonify({
        "prophet_forecast": 0,
        "xgboost_forecast": 0
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)


#=========================
# 주/월 학습 및 예측 API
#=========================

# 2-1. 주간 예측 (특정 날짜 1개)
@app.route('/predict/weekly/<int:store_id>', methods=['GET'])
def predict_weekly(store_id):
    date = request.args.get('date')
    if not date:
        return "Bad Request: date query parameter is required", 400

    print(f"📌 [주간 예측 요청] 매장 ID: {store_id}, 날짜: {date}")
    # 실제 Prophet 예측 로직은 추후 연결
    return jsonify({
        "date": date,
        "prophet_forecast": 0  # Dummy 예측값
    }), 200

# 2-2. 월간 예측 (특정 날짜 1개)
@app.route('/predict/monthly/<int:store_id>', methods=['GET'])
def predict_monthly(store_id):
    date = request.args.get('date')
    if not date:
        return "Bad Request: date query parameter is required", 400

    print(f"📌 [월간 예측 요청] 매장 ID: {store_id}, 날짜: {date}")
    # 실제 Prophet 예측 로직은 추후 연결
    return jsonify({
        "date": date,
        "prophet_forecast": 0  # Dummy 예측값
    }), 200

# ========================
# 서버 실행
# ========================
if __name__ == '__main__':
    app.run(debug=True, port=5001)
