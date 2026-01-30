import json


class Config:
    def __init__(self, filepath="config.json"):
        with open(filepath, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(self, k, v)  # convert dict key to attribute


config = Config()
