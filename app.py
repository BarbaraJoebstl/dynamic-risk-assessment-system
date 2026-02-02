from flask import Flask, session, jsonify, request, Response
import numpy as np
import pandas as pd
import json
import os
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, missing_data
from scoring import score_model
from logger_config import logger

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])

prediction_model = None


def convert_to_native(obj):
    if isinstance(obj, list):
        return [convert_to_native(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def get_prediction():
    # call the prediction function you created in Step 3
    try:
        data = request.get_json()
        if "dataset" not in data:
            return jsonify({"error": "please send the .csv for the prediction"}), 400

        path = data["dataset"]
        df = pd.read_csv(path)

        pred = model_predictions(df)
        pred = [int(p) if isinstance(p, (np.integer,)) else float(p) for p in pred]

        logger.info("Returning prediction")
        return jsonify({"predictions": pred}), 200
    except Exception as e:
        logger.error(f"/prediction error: {e}")
        return jsonify({"error": str(e)}), 500


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def get_latest_score():
    # check the score of the deployed model
    try:
        score = score_model()
        # returns value (a single F1 score number)
        return jsonify({"f1_score": score}), 200
    except Exception as e:
        logger.error(f"Error on `scoring` endpoint: {e}")
        return jsonify({"error": str(e)}), 500


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def get_summarystats():
    # check means, medians, and modes for each column

    try:
        stats = dataframe_summary()
        # returns a list of all calculated summary statistics
        return jsonify({"stats": stats}), 200
    except Exception as e:
        logger.error(f"Error on `get_summarystats` endpoint: {e}")
        return jsonify({"error": str(e)}), 500


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def get_diagnostics():
    # check timing and percent NA values
    try:
        timing = execution_time()
        na_values = missing_data()
        outdated_packages = outdated_packages_list()

        response_dict = {
            "execution_time_seconds": convert_to_native(timing),
            "missing_data": convert_to_native(na_values),
            "outdated_packages": convert_to_native(outdated_packages),
        }

        return Response(json.dumps(response_dict, indent=2), mimetype="application/json", status=200)
    except Exception as e:
        logger.error(f"Error on `diagnostics` endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
