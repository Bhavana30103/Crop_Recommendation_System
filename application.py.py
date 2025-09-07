from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

# ---------------------------
# Step 1: Initialize Flask App
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Step 2: Load Trained Model
# ---------------------------
model_path = "crop_recommendation_best.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"⚠️ Model file not found: {model_path}\n"
        f"👉 Run main.py first to generate crop_recommendation_best.pkl"
    )

model = joblib.load(model_path)
print(f"✅ Model loaded successfully from {os.path.abspath(model_path)}"  )

# ---------------------------
# Step 3: Routes
# ---------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = {f: float(request.form[f]) for f in features}

        # Convert to DataFrame (pipeline expects DataFrame input)
        X = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(X)[0]

        return render_template("index.html", prediction_text=f"Recommended Crop: {prediction}")

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------------
# Step 4: Run Flask App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
