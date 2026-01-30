import os
from config import config
from logger_config import logger
import shutil


####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    try:
        logger.info("Copying files")

        # Get source file paths
        model_file = os.path.join(config.output_model_path, "trainedmodel.pkl")
        score_file = os.path.join(config.output_model_path, "latestscore.txt")
        ingested_file = os.path.join(config.output_folder_path, "ingestedfiles.txt")

        files_to_copy = [model_file, score_file, ingested_file]

        # Ensure all files exist
        for f in files_to_copy:
            if not os.path.exists(f):
                logger.error(f"File not found: {f}")
                return

        # Ensure production deployment path exists
        os.makedirs(config.prod_deployment_path, exist_ok=True)

        # Copy files
        for f in files_to_copy:
            dest = os.path.join(config.prod_deployment_path, os.path.basename(f))
            shutil.copy(f, dest)
            logger.info(f"Copied {f} to {dest}")

        logger.info("Copy step completed successfully")

    except Exception as e:
        logger.exception(f"Copy to deployment folder failed: {e}")


if __name__ == "__main__":
    store_model_into_pickle()
