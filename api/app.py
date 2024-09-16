import logging

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn import metrics
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

df = pd.read_csv("./dataset/Crop_recommendation.csv")

model = joblib.load("./model/agroplan_model.joblib")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


def deviatedornot(data, val):
    variance = np.var(data)
    squared_difference1 = (val - np.mean(data)) ** 2
    squared_difference2 = (np.mean(data) - val) ** 2

    if (squared_difference1 > variance) & (squared_difference2 > variance):
        return False
    else:
        return True


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
        selected_crop = data["selected_crop"]
        N_leafSap = float(data["N_leafSap"])
        P_leafSap = float(data["P_leafSap"])
        K_leafSap = float(data["K_leafSap"])

        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )

        print("Input Parameter:")
        print(input_data)

        prediction = model.predict(input_data)
        crop_prediction = prediction[0]

        predicted_values = model.predict(X_test)
        knn_accuracy = metrics.accuracy_score(y_test, predicted_values) * 100

        knn_train_accuracy = model.score(X_train, y_train) * 100
        knn_test_accuracy = model.score(X_test, y_test) * 100

        print(f"\nKNN Accuracy is: {knn_accuracy:.2f} %")
        print(f"knn_train_accuracy = {knn_train_accuracy:.2f} %")
        print(f"knn_test_accuracy = {knn_test_accuracy:.2f} %")

        print(f"\n\nCrops Prediction (KNN Classifier): {crop_prediction}")

        x_rows = df[df["label"] == crop_prediction].drop(["label"], axis=1)
        print(x_rows)

        mean_val = [x_rows[col].mean() for col in x_rows.columns]
        recommendations = {}
        percent = []

        for x, y in zip(mean_val, x_rows.columns):
            sensor_value = input_data[y].values[0]
            recommendation = round(x - sensor_value, 2)

            if y in ["N", "P", "K"]:
                print(
                    f"Nutrisi {y} nilai Standart (Rerata): {round(x,2)} dan Data Sensor: {round(sensor_value,2)} ==>> Rekomendasi: {recommendation}"
                )
            elif y == "humidity":
                print(
                    f"{y} (RH/kelembaban) rata-rata : {round(x,2)} dan Data Sensor: {round(sensor_value,2)} ==>> Rekomendasi: {recommendation}"
                )
            elif y == "ph":
                print(
                    f"{y} (Potential of Hydrogen) rata-rata: {round(x,2)} dan Data Sensor: {round(sensor_value,2)} ==>> Rekomendasi: {recommendation}"
                )
            else:
                print(
                    f"{y} nilai Standart (Rerata): {round(x,2)} dan Data Sensor: {round(sensor_value,2)} ==>> Rekomendasi: {recommendation}"
                )

            recommendations[y] = recommendation

            if x > sensor_value:
                percent.append(-((x - sensor_value) * 100 / x))
            elif x <= sensor_value:
                percent.append(((sensor_value - x) * 100 / sensor_value))

        if crop_prediction == selected_crop:
            crop_check_result = f"The current state of soil is {selected_crop} ready"
        else:
            crop_check_result = f"{selected_crop} can't be planted in such conditions"

        comparisons = {}
        comparisons_value = {}
        nutrient_values = {"N": N, "P": P, "K": K}
        predictions = {
            "N": recommendations["N"],
            "P": recommendations["P"],
            "K": recommendations["K"],
        }
        leafSap = [N_leafSap, P_leafSap, K_leafSap]

        for nutrient in ["N", "P", "K"]:
            actual_value = nutrient_values[nutrient]
            predicted_value = predictions[nutrient]
            leafSap_value = leafSap[["N", "P", "K"].index(nutrient)]
            difference = round(predicted_value - actual_value - leafSap_value, 2)

            logging.debug(
                f"Actual value: {actual_value}, Predicted value: {predicted_value}, Difference: {difference}, Leaf Sap: {leafSap_value}"
            )

            if difference > 0:
                comparisons_value[nutrient] = difference
                comparisons[nutrient] = f"Sufficient (Surplus = {difference:.2f})"
            elif difference < 0:
                comparisons_value[nutrient] = difference
                comparisons[nutrient] = f"Insufficient (Deficit = -{-difference:.2f})"
            else:
                comparisons[nutrient] = "Sufficient"

        response = {
            "knn_accuracy": round(knn_accuracy, 2),
            "knn_train_accuracy": round(knn_train_accuracy, 2),
            "knn_test_accuracy": round(knn_test_accuracy, 2),
            "crop_prediction": crop_prediction,
            "selected_crop": selected_crop,
            "recommendations": recommendations,
            "crop_check_result": crop_check_result,
            "percent_differences": percent,
            "comparisons": comparisons,
            "comparisons_value": comparisons_value,
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
