import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = None
x_test = None
y_test = None
x_train = None
y_train = None
crops = None


def load_data_and_train_model():
    global model, x_test, y_test, x_train, y_train, crops
    
    crops = pd.read_csv("./dataset/Crop_recommendation.csv")
    features = crops[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    target = crops["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=2
    )

    model = KNeighborsClassifier()
    model.fit(x_train, y_train)

    print("Model trained and ready.")


# Call this function when the app starts
load_data_and_train_model()


@app.route("/api/predict", methods=["POST"])
def predict():
    global model, x_test, y_test, x_train, y_train, crops

    try:
        # Get input data from request
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

        x_values = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        x_data = pd.DataFrame(
            x_values,
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )
        print("Input Parameter:\n", x_data)

        prediction = model.predict(x_data)
        crop_prediction = prediction[0]

        predicted_values = model.predict(x_test)
        knn_accuracy = metrics.accuracy_score(y_test, predicted_values) * 100

        # Train and test accuracy
        knn_train_accuracy = model.score(x_train, y_train) * 100
        knn_test_accuracy = model.score(x_test, y_test) * 100

        print(f"\nKNN Accuracy is: {knn_accuracy:.2f} %")

        print(f"\n\nCrops Prediction (KNN Classifier): {crop_prediction}")

        crop_data = crops[crops["label"] == selected_crop]
        if crop_data.empty:
            return jsonify({"error": "Crop not found in dataset"}), 400

        # Load the full dataset again to get the mean values for the predicted crop
        df = pd.read_csv("./dataset/Crop_recommendation.csv")
        x_rows = df[df["label"] == crop_prediction].drop(["label"], axis=1)
        print(x_rows)

        mean_values = {
            col: crop_data[col].mean()
            for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        }

        # mean values 
        mean_N = mean_values["N"]
        mean_P = mean_values["P"]
        mean_K = mean_values["K"]
        mean_temperature = mean_values["temperature"]
        mean_humidity = mean_values["humidity"]
        mean_ph = mean_values["ph"]
        mean_rainfall = mean_values["rainfall"]

        recommendations = {}
        percent = []

        for y, x in mean_values.items():
            sensor_value = x_data[y].values[0]
            recommendation = round(x - sensor_value, 2)

            if y in ["N", "P", "K"]:
                print(
                    f"Nutrisi {y} nilai Standart (Rerata): {round(x, 2)} dan Data Sensor: {round(sensor_value, 2)} ==>> Rekomendasi: {recommendation}"
                )
            elif y == "humidity":
                print(
                    f"{y} (RH/kelembaban) rata-rata : {round(x, 2)} dan Data Sensor: {round(sensor_value, 2)} ==>> Rekomendasi: {recommendation}"
                )
            elif y == "ph":
                print(
                    f"{y} (Potential of Hydrogen) rata-rata: {round(x, 2)} dan Data Sensor: {round(sensor_value, 2)} ==>> Rekomendasi: {recommendation}"
                )
            else:
                print(
                    f"{y} nilai Standart (Rerata): {round(x, 2)} dan Data Sensor: {round(sensor_value, 2)} ==>> Rekomendasi: {recommendation}"
                )

            recommendations[y] = recommendation

            if x > sensor_value:
                percent.append(-((x - sensor_value) * 100 / x))
            elif x <= sensor_value:
                percent.append(((sensor_value - x) * 100 / sensor_value))

        comparisons = {}
        comparisons_value = {}
        nutrient_values = {"N": N, "P": P, "K": K}
        predictions = {
            "N": round(mean_N, 2),
            "P": round(mean_P, 2),
            "K": round(mean_K, 2),
        }
        leafSap = {"N": N_leafSap, "P": P_leafSap, "K": K_leafSap}
        
        for nutrient in ["N", "P", "K"]:
            envir_value = nutrient_values[nutrient]
            leaf_sap_value = leafSap[nutrient]
            predict_value = predictions[nutrient]
            print(envir_value, leaf_sap_value,predict_value )
            recommend_value = (envir_value + leaf_sap_value) - predict_value

            comparisons_value[nutrient] = round(recommend_value, 2)

            if recommend_value > 0:
                comparisons[nutrient] = f"Sufficient (Surplus = {recommend_value:.2f})"
            elif recommend_value < 0:
                comparisons[nutrient] = (
                    f"Insufficient (Deficit = -{-recommend_value:.2f})"
                )
            else:
                comparisons[nutrient] = "Sufficient"

        if crop_prediction == selected_crop:
            crop_check_result = f"The current state of soil is {selected_crop} ready"
        else:
            crop_check_result = f"{selected_crop} can't be planted in such conditions"

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
            "mean_values":  {
                "N": round(mean_N, 2),
                "P": round(mean_P, 2),
                "K": round(mean_K, 2),
                "temperature": round(mean_temperature, 2),
                "humidity": round(mean_humidity, 2),
                "ph": round(mean_ph, 2),
                "rainfall": round(mean_rainfall, 2),
            },
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)