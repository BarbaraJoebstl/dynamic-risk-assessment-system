import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from diagnostics import model_predictions
import matplotlib.pyplot as plt
import os
from config import config
from logger_config import logger


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    # read the deployed model and a test dataset, calculate predictions
    logger.info("Starting to generate confusion matrix")
    test_data_set = os.path.join(config.test_data_path, "testdata.csv")
    df = pd.read_csv(test_data_set)

    y_pred = model_predictions(df)
    # Get true labels (pandas Series)
    y_test = df[config.target]

    cm = confusion_matrix(y_test, y_pred)

    # Plot using seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    output_file = os.path.join(config.output_model_path, "confusionmatrix.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Confusion matrix saved to {output_file}")


if __name__ == "__main__":
    score_model()
