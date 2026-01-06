from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# -------------------------------
# Load model + label encoder
# -------------------------------
model = joblib.load("model.pkl")
smoke_encoder = joblib.load("label_encoder.pkl")

print("Model and Encoder loaded successfully!")


# -------------------------------
# Simple health check
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Diabetes Prediction API is running"}), 200


# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        # Required fields
        required = [
            "age","hypertension","heart_disease",
            "smoking_history","bmi","HbA1c_level","blood_glucose_level"
        ]

        # Check missing fields
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Encode smoking_history
        try:
            smoking_val = smoke_encoder.transform([data["smoking_history"]])[0]
        except:
            return jsonify({
                "error": "Invalid smoking_history value",
                "allowed": list(smoke_encoder.classes_)
            }), 400

        # Build model input
        input_array = np.array([[
            float(data["age"]),
            float(data["hypertension"]),
            float(data["heart_disease"]),
            smoking_val,
            float(data["bmi"]),
            float(data["HbA1c_level"]),
            float(data["blood_glucose_level"])
        ]])

        pred = int(model.predict(input_array)[0])
        result = "Diabetes" if pred == 1 else "No Diabetes"

        return jsonify({
            "prediction": pred,
            "result": result
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Server error",
            "detail": str(e)
        }), 500


# -------------------------------
# Run locally
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
