import pandas as pd
import timeit
import os
import pickle
from sklearn import metrics
from config import config
from logger_config import logger
import subprocess


##################Function to get model predictions
def model_predictions(df):
    # read the deployed model and a test dataset, calculate predictions
    model_file = os.path.join(config.output_model_path, config.model_file)
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    df.drop(config.uneeded_cols, axis="columns", inplace=True)

    # Features and target
    X_test = df[config.features]
    y_test = df[config.target]

    y_pred = model.predict(X_test)

    # check length
    if len(y_pred) != len(df):
        alert_msg = f"Prediction length mismatch: {len(y_pred)} predictions for {len(df)} rows"
        logger.error(alert_msg)
        raise ValueError(alert_msg)

    # Calculate metrics
    f1 = metrics.f1_score(y_test, y_pred)
    score_file = os.path.join(config.output_model_path, "latestscore.txt")
    with open(score_file, "w") as f:
        f.write(str(f1))
    logger.info(f"Saved F1 {f1:.6f} score to: {score_file}")

    # returns value should be a list containing all predictions
    return list(y_pred)


##################Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    path = os.path.join(config.output_folder_path, config.ingested_data)
    df = pd.read_csv(path)
    # only numeric atm
    df = df[config.features]

    na_percentages = (df.isna().sum() / len(df)) * 100

    summary = []

    for col, pct in na_percentages.items():
        summary.append({"column": col, "na_percent": pct})

    logger.info(f"NAs for every col: {summary}")
    # returns list containing all summary statistics
    return summary


##################Missing Data checks
def missing_data():
    data_path = os.path.join(config.output_folder_path, "finaldata.csv")
    df = pd.read_csv(data_path)

    # percentage of missing values per column
    na_percentages = (df.isna().sum() / len(df)) * 100

    missing_list = [{"column": col, "na_percent": float(pct)} for col, pct in na_percentages.items()]

    logger.info(f"NAs for every col: {missing_list}")
    return missing_list


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    logger.info("Getting execution time")
    timings = {}
    start_i = timeit.default_timer()
    os.system("python ingestion.py")
    ingestion_time = timeit.default_timer() - start_i
    timings["ingestion_time"] = ingestion_time
    logger.info(f"Ingestion time: {timings['ingestion_time']}")

    start_t = timeit.default_timer()
    os.system("python training.py")
    training_time = timeit.default_timer() - start_t
    timings["training_time"] = training_time
    logger.info(f"Training time: {timings['training_time']}")

    # returns a list of 2 timing values in seconds
    return timings


##################Function to check dependencies
def outdated_packages_list():
    # get a list of
    logger.info("Checking for outdated packages")
    # run pip list --outdated (default table format)
    result = subprocess.run(["pip", "list", "-o"], capture_output=True, text=True)

    lines = result.stdout.strip().split("\n")

    outdated_packages = []

    for line in lines[2:]:
        if line.strip():
            parts = line.split()
            # parts: [Package, Version, Latest, Type]
            if len(parts) >= 3:
                outdated_packages.append({"package": parts[0], "current": parts[1], "latest": parts[2]})

    logger.info(f"Found {len(outdated_packages)} outdated packages:")
    logger.info(f"{outdated_packages}")
    return outdated_packages


if __name__ == "__main__":
    test_data_set = os.path.join(config.output_folder_path, "finaldata.csv")
    df = pd.read_csv(test_data_set)
    model_predictions(df)
    dataframe_summary()
    execution_time()
    outdated_packages_list()
