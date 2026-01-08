from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ---------------------------------
# Load model and label encoder
# ---------------------------------
try:
    model = joblib.load("model.pkl")
    smoke_encoder = joblib.load("label_encoder.pkl")
    print("✅ Model and Encoder loaded successfully")
except Exception as e:
    print("❌ Failed to load model files:", e)
    raise e


# ---------------------------------
# Health check route (IMPORTANT)
# ---------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Diabetes Prediction API is running"
    }), 200


# ---------------------------------
# Prediction route
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        required_fields = [
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]

        # Validate required fields
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Encode smoking history
        try:
            smoking_encoded = smoke_encoder.transform(
                [data["smoking_history"]]
            )[0]
        except Exception:
            return jsonify({
                "error": "Invalid smoking_history value",
                "allowed_values": list(smoke_encoder.classes_)
            }), 400

        # Prepare input for model
        input_data = np.array([[
            float(data["age"]),
            float(data["hypertension"]),
            float(data["heart_disease"]),
            smoking_encoded,
            float(data["bmi"]),
            float(data["HbA1c_level"]),
            float(data["blood_glucose_level"])
        ]])

        prediction = int(model.predict(input_data)[0])
        result = "Diabetes" if prediction == 1 else "No Diabetes"

        return jsonify({
            "prediction": prediction,
            "result": result
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "detail": str(e)
        }), 500


# ---------------------------------
# App entry point (Render compatible)
# ---------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
