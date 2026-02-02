import pandas as pd
import pickle
import os
from sklearn import metrics
from logger_config import logger
from config import config


#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    try:
        logger.info("Starting model scoring")

        test_file = os.path.join(config.test_data_path, "testdata.csv")  # adjust filename
        if not os.path.exists(test_file):
            logger.error(f"Test data not found at {test_file}")
            return

        test_df = pd.read_csv(test_file)
        # drop unused columns
        test_df.drop(config.uneeded_cols, axis="columns", inplace=True)
        logger.info(f"Loaded test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

        # Features and target
        X_test = test_df[config.features]
        y_test = test_df[config.target]

        # Load trained model
        model_file = os.path.join(config.output_model_path, config.model_file)
        if not os.path.exists(model_file):
            logger.error(f"Trained model not found at {model_file}")

        with open(model_file, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded trained model")

        y_pred = model.predict(X_test)

        # Calculate F1 score and store
        f1 = metrics.f1_score(y_test, y_pred)
        score_file = os.path.join(config.output_model_path, "latestscore.txt")
        with open(score_file, "w") as f:
            f.write(str(f1))
        logger.info(f"Saved F1 {f1:.6f} score to: {score_file}")
        return f1

    except Exception as e:
        logger.exception(f"Model scoring failed: {e}")


if __name__ == "__main__":
    score_model()
