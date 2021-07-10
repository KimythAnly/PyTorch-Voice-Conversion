import yaml

def load_config(path):
    config = yaml.safe_load(open(path, 'r'))
    return config