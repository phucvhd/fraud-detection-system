import os
import yaml


def _get_project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "config")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Could not find project root (no 'config/' folder found).")


class ConfigLoader:
    def __init__(self, path="application.yaml"):
        self.path = path
        self.config = self._load_config()

    def _load_config(self):
        root_dir = _get_project_root()
        config_path = os.path.join(root_dir, "config", self.path)

        if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def as_dict(self):
        return self.config
