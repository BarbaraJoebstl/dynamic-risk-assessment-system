import requests
from config import config
import os
import json

# Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000"

test_data_set = os.path.join(config.input_folder_path, "finaldata.csv")


# Call each API endpoint and store the responses
response1 = requests.post(f"{URL}/prediction", json={"dataset": test_data_set})
response2 = requests.get(f"{URL}/scoring")
response3 = requests.get(f"{URL}/summarystats")
response4 = requests.get(f"{URL}/diagnostics")

# combine all API responses
responses = {
    "prediction": response1.json(),
    "scoring": response2.json(),
    "summary_stats": response3.json(),
    "diagnostics": response4.json(),
}

# write the responses to your workspace

output_file = os.path.join(config.output_model_path, "api_responses.json")

with open(output_file, "w") as f:
    json.dump(responses, f, indent=2)

print(f"API responses saved to {output_file}")
