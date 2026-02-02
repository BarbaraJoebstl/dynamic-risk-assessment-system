import pandas as pd
import os
from datetime import datetime, timezone
from logger_config import logger
from config import config


#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    try:
        dataframes = []
        file_list = []

        # ingest
        csv_files = [f for f in os.listdir(config.input_folder_path) if f.endswith(".csv")]
        for file in csv_files:
            file_path = os.path.join(config.input_folder_path, file)
            file_list.append(file)
            df = pd.read_csv(file_path)
            dataframes.append(df)

        # process
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df = merged_df.drop_duplicates().reset_index(drop=True)
        merged_df["ingestion_time"] = datetime.now(timezone.utc).isoformat()

        # store merged df
        os.makedirs(config.output_folder_path, exist_ok=True)
        output_file = os.path.join(config.output_folder_path, config.ingested_data)
        merged_df.to_csv(output_file, index=False)

        logger.info(f"Merged data written to: {output_file}")
        logger.info(f"Rows after deduplication: {len(merged_df)}")

        # store list of used files separate (requirement)
        files_used_txt = os.path.join(config.output_folder_path, "ingestedfiles.txt")
        with open(files_used_txt, "w") as f:
            f.write(str(file_list))
        logger.info(f"Ingested files recorded in: {files_used_txt}")

    except Exception as e:
        logger.exception(f"Data ingestion failed: {e}")


if __name__ == "__main__":
    merge_multiple_dataframe()
