import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_CONFIG_SENSITIVE_PATH = Path(__file__).parent / "configSensitive.yaml"
_COMMANDS_PATH = Path(__file__).parent / "commands_config.yaml"

# Internal cache so each file is only read once
_config_cache = None
_commands_cache = None


def _load_config():
    """Load the YAML config file into memory."""
    global _config_cache

    if _config_cache is None:
        if not _CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")

        if not _CONFIG_SENSITIVE_PATH.exists():
            raise FileNotFoundError(f"ConfigSensitive file not found: {_CONFIG_SENSITIVE_PATH}")

        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}

        with open(_CONFIG_SENSITIVE_PATH, "r", encoding="utf-8") as f:
            sensitive_config = yaml.safe_load(f) or {}

        _config_cache = base_config

        # Merge sensitive config into base config
        for category, keys in sensitive_config.items():
            if category in _config_cache and isinstance(_config_cache[category], dict) and isinstance(keys, dict):
                _config_cache[category].update(keys)
            else:
                _config_cache[category] = keys
            
    return _config_cache


def get_config():
    """
    Public accessor for configuration data.
    Returns the parsed YAML as a Python dict.
    """
    return _load_config()


def _load_commands_config():
    """Load the command-rule YAML file into memory."""
    global _commands_cache

    if _commands_cache is None:
        if not _COMMANDS_PATH.exists():
            raise FileNotFoundError(f"Commands config file not found: {_COMMANDS_PATH}")

        with open(_COMMANDS_PATH, "r", encoding="utf-8") as f:
            _commands_cache = yaml.safe_load(f) or {}

    return _commands_cache


def get_commands_config():
    """
    Public accessor for active-command functionality rules.
    Returns the parsed YAML as a Python dict.
    """
    return _load_commands_config()
