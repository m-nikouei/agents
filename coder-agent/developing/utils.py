import json

def read_configs(config_file):
    f = open(config_file)
    configs = json.load(f)
    f.close()
    return configs