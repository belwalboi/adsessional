from flask  import Flask, request, jsonify, send_from_directory
import pickle

app = Flask(__name__, static_folder="static")

# Load models
lr = pickle.load(open("model_linear.pkl","rb"))
rf = pickle.load(open("model_rf.pkl","rb"))
xgb = pickle.load(open("model_xgb.pkl","rb"))
vec = pickle.load(open("vectorizer.pkl","rb"))
borough_map = pickle.load(open("borough_map.pkl","rb"))

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    d = {
        "cluster_id": borough_map[data["borough"]],
        "Hour": data["hour"],
        "DayOfWeek": data["day"],
        "Month": data["month"],
        "temp": data["temp"],
        "spd": data["spd"],
        "vsb": data["vsb"],
        "pcp01": data["pcp"],
        "hday": data["hday"],
        "IsWeekend": 1 if data["day"] >= 5 else 0,
        "IsRushHour": 1 if data["hour"] in [7,8,9,17,18,19] else 0
    }

    X = vec.transform([d])

    return jsonify({
        "Linear Regression": float(lr.predict(X)[0]),
        "Random Forest": float(rf.predict(X)[0]),
        "XGBoost": float(xgb.predict(X)[0])
    })

if __name__ == "__main__":
    app.run()