from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# ========================
# 1. 학습 요청
# ========================

# 1-1. 주간 학습 (revenue_data만 받음)
@app.route('/train/weekly', methods=['POST'])
def train_weekly():
    data = request.get_json()
    revenue_data = data.get('revenue_data')

    if not revenue_data:
        return "Bad Request: revenue_data is missing", 400

    print("📌 [주간 학습 요청] 받은 매출 데이터:", revenue_data)
    # 실제 Prophet 학습 로직은 추후 연결
    return '', 200


# 1-2. 월간 학습 (revenue_data만 받음)
@app.route('/train/monthly', methods=['POST'])
def train_monthly():
    data = request.get_json()
    revenue_data = data.get('revenue_data')

    if not revenue_data:
        return "Bad Request: revenue_data is missing", 400

    print("📌 [월간 학습 요청] 받은 매출 데이터:", revenue_data)
    # 실제 Prophet 학습 로직은 추후 연결
    return '', 200

# ========================
# 2. 예측 요청
# ========================

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
