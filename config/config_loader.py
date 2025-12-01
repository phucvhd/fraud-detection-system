import os
import yaml
from dotenv import load_dotenv

def _get_project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "config")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Could not find project root (no 'config/' folder found).")


class ConfigLoader:
    def __init__(self, path="application.yaml"):
        load_dotenv()
        self.path = path
        self.config = self._load_config()

    def _load_config(self):
        root_dir = _get_project_root()
        config_path = os.path.join(root_dir, "config", self.path)

        if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return self._substitute_env_vars(config)

    def _substitute_env_vars(self, obj):
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_name = obj[2:-1]
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Environment variable '{var_name}' not found!")
            return value
        return obj

    def get(self, key, default=None):
        return self.config.get(key, default)

    def as_dict(self):
        return self.config