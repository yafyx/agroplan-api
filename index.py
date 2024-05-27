import logging
from http.server import BaseHTTPRequestHandler
from os.path import join

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

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
        float(data["pH"]),
        float(data["Temperature"]),
        float(data["Humidity"]),
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
        input_features = [
            nutrient_values[n] for n in ["N", "P", "K"] if n != nutrient
        ] + common_features
        logging.debug(f"Input features for {nutrient} prediction: {input_features}")
        prediction = models[nutrient].predict([input_features])[0]
        logging.debug(f"Prediction for {nutrient}: {prediction}")
        predictions[nutrient] = prediction

    comparisons = {
        "N": "Sufficient"
        if float(data["N"]) >= predictions["N"]
        else "Insufficient, needs to be increased",
        "P": "Sufficient"
        if float(data["P"]) >= predictions["P"]
        else "Insufficient, needs to be increased",
        "K": "Sufficient"
        if float(data["K"]) >= predictions["K"]
        else "Insufficient, needs to be increased",
    }

    response = {"predictions": predictions, "comparisons": comparisons}

    return jsonify(response)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)

        with app.test_client() as client:
            response = client.post(
                "/api/predict", data=body, content_type="application/json"
            )
            self.send_response(response.status_code)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(response.get_data())

    def do_GET(self):
        self.send_response(405)
        self.end_headers()


# The following line is only needed if you want to test the server locally
# if __name__ == "__main__":
#     app.run(port=5328)
