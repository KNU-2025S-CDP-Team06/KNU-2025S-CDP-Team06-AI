from flask import Flask, request, jsonify

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