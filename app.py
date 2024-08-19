import logging
from os.path import join

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)

models = {}
for nutrient in ["N", "P", "K"]:
    model_path = join("data", f"model_{nutrient}.pkl")
    logging.debug(f"Loading model from {model_path}")
    with open(model_path, "rb") as file:
        models[nutrient] = joblib.load(file)
    logging.debug(f"Loaded model for {nutrient}")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    common_features = [
        float(data["Temperature"]),
        float(data["Humidity"]),
        float(data["pH"]),
        float(data["Rainfall"]),
        int(data["Crop"]),
    ]

    nutrient_values = {
        "N": float(data["N"]),
        "P": float(data["P"]),
        "K": float(data["K"]),
    }

    logging.debug(f"Received input features: {nutrient_values} and {common_features}")

    predictions = {}
    for nutrient in ["N", "P", "K"]:
        if nutrient == "N":
            input_features = [nutrient_values[n] for n in ["P", "K"]] + common_features
        elif nutrient == "P":
            input_features = [nutrient_values[n] for n in ["N", "K"]] + common_features
        elif nutrient == "K":
            input_features = [nutrient_values[n] for n in ["N", "P"]] + common_features

        logging.debug(f"Input features for {nutrient} prediction: {input_features}")
        prediction = models[nutrient].predict([input_features])[0]
        logging.debug(f"Prediction for {nutrient}: {prediction}")
        predictions[nutrient] = prediction

    comparisons = {}
    for nutrient in ["N", "P", "K"]:
        actual_value = nutrient_values[nutrient]
        predicted_value = predictions[nutrient]
        difference = round(predicted_value - actual_value , 2)
        logging.debug(
            f"Actual value: {actual_value}, Predicted value: {predicted_value}, Difference: {difference}"
        )

        if difference > 0:
            comparisons[nutrient] = f"Sufficient (Surplus = {difference:.2f})"
        elif difference < 0:
            comparisons[nutrient] = f"Insufficient (Deficit = -{-difference:.2f})"
        else:
            comparisons[nutrient] = "Sufficient"

    response = {"predictions": predictions, "comparisons": comparisons}
    logging.debug(f"Response: {response}")

    return jsonify(response)


if __name__ == "__main__":
    app.run(port=5328)
