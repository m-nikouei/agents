# config.py

import json

def read_configs(config_file):
    with open(config_file) as f:
        configs = json.load(f)
    return configs