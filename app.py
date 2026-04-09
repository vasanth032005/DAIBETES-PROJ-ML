import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from flask import Flask, request, jsonify

# -------------------------------
# Step 1: Load dataset
# -------------------------------
dataset_file = 'diabetes.csv'

try:
    df = pd.read_csv(dataset_file)
    print(f"Dataset loaded successfully from: {dataset_file}")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Dataset file not found: {dataset_file}")
    exit()

# -------------------------------
# Step 2: Prepare data
# -------------------------------
if 'Outcome' in df.columns:
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
else:
    print("Error: 'Outcome' column not found in dataset.")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 3: Train model
# -------------------------------
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# -------------------------------
# Step 4: Save model
# -------------------------------
model_filename = 'logistic_regression_model.joblib'
joblib.dump(model, model_filename)

# -------------------------------
# Step 5: Flask App
# -------------------------------
app = Flask(__name__)

# Load model
loaded_model = joblib.load(model_filename)

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Diabetes Prediction API!",
        "required_features": X.columns.tolist()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        input_df = pd.DataFrame([data])
        input_df = input_df[X.columns]

        prediction = loaded_model.predict(input_df)
        probability = loaded_model.predict_proba(input_df)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability_class_0": float(probability[0][0]),
            "probability_class_1": float(probability[0][1])
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing feature: {e}",
            "required_features": X.columns.tolist()
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500
