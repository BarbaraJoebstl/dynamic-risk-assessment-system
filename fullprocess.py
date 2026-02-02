from ingestion import merge_multiple_dataframe
import os
from config import config
from logger_config import logger
import pickle
import pandas as pd
from sklearn import metrics
import subprocess
import sys
import ast


def load_source_data():
    ##################Check and read new data
    # first, read ingestedfiles.txt
    merge_multiple_dataframe()
    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    # get filenames in sourcedata folder
    source_csv_files = [f for f in os.listdir(config.input_folder_path) if f.endswith(".csv")]

    # get filenames form ingestedfiles.txt
    with open(os.path.join(config.output_folder_path, "ingestedfiles.txt"), "r") as f:
        line = f.read().strip()
        # convert the string representation of a list into an actual Python list
        ingested_files = ast.literal_eval(line)
    # Determine new/untracked files
    ##################Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process here
    if set(source_csv_files) != set(ingested_files):
        logger.error("There is a missmatch between the files to use and the used ones. Please check your code.")
        raise Exception
    else:
        logger.info("All needed data read in. Contiuning")


def check_for_model_drift_and_report():
    ##################Checking for model drift
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    # 1. get the latestscore stored in latestscore
    logger.info("get latest score")

    with open(os.path.join(config.output_model_path, "latestscore.txt"), "r") as f:
        score_str = f.read().strip()  # read as string
        old_f1 = float(score_str)
        logger.info(f"old F1: {old_f1}")
    # 2. calc F1 on the new data with that model
    logger.info("load existing model")
    with open(os.path.join(config.output_model_path, config.trained_model), "rb") as f:
        model = pickle.load(f)

    data = os.path.join(config.output_folder_path, "finaldata.csv")
    df = pd.read_csv(data)
    df.drop(config.uneeded_cols, axis="columns", inplace=True)
    # Features and target
    X_test = df[config.features]
    y_test = df[config.target]
    y_pred = model.predict(X_test)

    new_f1 = metrics.f1_score(y_test, y_pred)
    logger.info(f"new f1: {new_f1}")
    # 3. check for model drift
    ##################Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here
    if new_f1 < old_f1:
        logger.error("model drift detected. Please check the data")
        raise Exception
    ##################Re-deployment
    # if you found evidence for model drift, re-run the deployment.py script
    else:
        # retrain and deploy
        subprocess.run([sys.executable, "training.py"], capture_output=True, text=True)
        subprocess.run([sys.executable, "deployment.py"], capture_output=True, text=True)

        ##################Diagnostics and reporting
        # run diagnostics.py and reporting.py for the re-deployed model
        logger.info("running diagnostics ans reporting")
        subprocess.run([sys.executable, "diagnostics.py"], capture_output=True, text=True)
        subprocess.run([sys.executable, "reporting.py"], capture_output=True, text=True)


if __name__ == "__main__":
    load_source_data()
    check_for_model_drift_and_report()
