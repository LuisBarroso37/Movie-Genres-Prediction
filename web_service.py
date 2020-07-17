"""
web_service.py.

This script runs a flask web service.
"""

# Imports
from flask import Flask, jsonify, request, make_response
from .model import Model
import pandas as pd

app = Flask(__name__)

# Initialize Model object
model = Model()


@app.route("/genres/train", methods=["POST"])
def train_model():
    """
    Use method for flask web service in route /genres/train.

    Trains the model.
    """
    # Decode the request
    data = request.data.decode("utf-8")

    # Write data from the request in a local csv file
    train_csv = "train_local.csv"
    f = open(train_csv, "w", encoding="utf-8")
    f.write(data)
    f.close()

    # Load the train csv file as a DataFrame
    train_df = pd.read_csv(train_csv)

    # Train model
    model.train_model(train_df)

    return jsonify({"success": "The model was trained sucessfully"})


@app.route("/genres/predict", methods=["POST"])
def predict_model():
    """
    Use method for flask web service in route /genres/predict.

    Makes predictions.
    """
    # Decode the request
    data = request.data.decode("utf-8")

    # Write data from the request in a local csv file
    test_csv = "test_local.csv"
    f = open(test_csv, "w", encoding="utf-8")
    f.write(data)
    f.close()

    # Load the test csv file as a DataFrame
    test_df = pd.read_csv(test_csv)

    # Get submission DataFrame
    predictions_df = model.predict(test_df)

    # Send csv file as response
    res = make_response(predictions_df.to_csv(index=False))
    res.headers["Content-Disposition"] = "attachment; filename=submission.csv"
    res.headers["Content-Type"] = "text/csv"
    return res


if __name__ == "__main__":
    app.run()
