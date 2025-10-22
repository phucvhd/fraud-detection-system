import os
import yaml


def load_config(path="config.yaml"):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ConfigLoader:
    def __init__(self, path="application.yaml"):
        self.path = path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Config file not found: {self.path}")
        with open(self.path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def as_dict(self):
        return self.config
