import logging
from os.path import join

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

df = pd.read_csv("./dataset/Crop_recommendation.csv")

model = joblib.load("./model/agroplan_model.joblib")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )

        print("Input Parameter:")
        print(input_data)

        prediction = model.predict(input_data)
        crop_prediction = prediction[0]

        knn_train_accuracy = model.score(X_train, y_train) * 100
        knn_test_accuracy = model.score(X_test, y_test) * 100

        print(f"\nKNN Accuracy is: {knn_test_accuracy:.2f} %")
        print(f"knn_train_accuracy = {knn_train_accuracy:.2f} %")
        print(f"knn_test_accuracy = {knn_test_accuracy:.2f} %")

        print(f"\n\nCrops Prediction (KNN Classifier): {crop_prediction}")

        crop_data = df[df["label"] == crop_prediction]
        averages = crop_data.mean()

        recommendations = {}
        for nutrient in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            std_value = averages[nutrient]
            sensor_value = input_data[nutrient].values[0]
            recommendation = round(std_value - sensor_value, 2)
            print(
                f"{nutrient} nilai Standart (Rerata): {std_value:.2f} dan Data Sensor: {sensor_value} ==>> Rekomendasi: {recommendation}"
            )
            recommendations[nutrient] = recommendation

        if crop_prediction.lower() == "rice":
            crop_check_result = "rice can be planted in such conditions"
        else:
            crop_check_result = "rice can't be planted in such conditions"

        print(f"\n{crop_check_result}\n")

        response = {
            "knn_train_accuracy": round(knn_train_accuracy, 2),
            "knn_test_accuracy": round(knn_test_accuracy, 2),
            "crop_prediction": crop_prediction,
            "recommendations": recommendations,
            "crop_check_result": crop_check_result,
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
