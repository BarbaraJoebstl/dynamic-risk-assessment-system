import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression

from logger_config import logger
from config import config


#################Function for training the model
def train_model():
    """
    About the data trained here:
    One row represents a hypothetical corporation. There are five columns in the dataset:

    "corporation", which contains four-character abbreviations for names of corporations
    "lastmonth_activity", which contains the level of activity associated with each corporation over the previous month
    "lastyear_activity", which contains the level of activity associated with each corporation over the previous year
    "number_of_employees", which contains the number of employees who work for the corporation
    "exited", which contains a record of whether the corporation exited their contract (1 indicates that the corporation exited, and 0 indicates that the corporation did not exit)
    The dataset's final column, "exited", is the target variable for our predictions. The first column, "corporation", will not be used in modeling. The other three numeric columns will all be used as predictors in your ML model.


    corporation	identifier (not used for modeling) - 4 char abbr for corporations
    lastmonth_activity	numeric predictor
    lastyear_activity	numeric predictor
    number_of_employees	numeric predictor
    exited	target variable (0 = stayed, 1 = exited)
    ingestion_time timestamp, not used for training
    """

    logger.info("Starting model training")
    # get te data
    data_file = os.path.join(config.output_folder_path, "finaldata.csv")
    df = pd.read_csv(data_file)

    # drop unneded cols
    df.drop(["corporation", "ingestion_time"], axis="columns", inplace=True)
    logger.info(f"Loaded training data, removed unuesed cols: {df.shape[0]} rows, {df.shape[1]} columns")

    # set features and the target - no need to split the data for this assignement, we take as is.
    X = df[config.features]
    y = df[config.target]

    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    model.fit(X, y)
    # write the trained model to your workspace in a file called trainedmodel.pkl
    model_file = os.path.join(config.output_model_path, config.model_file)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
        logger.info(f"Trained model saved to: {model_file}")


if __name__ == "__main__":
    train_model()
