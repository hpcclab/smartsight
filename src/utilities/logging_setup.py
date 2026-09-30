"""SmartSight loggers and the optional per-run log folder."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_HANDLER_MARK = "_smartsight_run_log"
_SECRET_KEY = re.compile(
    r"(api[_-]?key|token|secret|password|credential|authorization)",
    re.IGNORECASE,
)
_UNSAFE_PATH = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def get_logger(name: str) -> logging.Logger:
    """Return a SmartSight logger that does not inherit the root level.

    Importing Paddle sets the root logger to WARNING after the app has
    configured it. A NOTSET child inherits that level and drops INFO.
    An explicit INFO level still propagates to the root stderr handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def configure_file_logging(config, started_at=None, logs_root=None):
    """Write this process's logs into a new run folder, once.

    The folder is ``{logs_root}/{name}_{description}_{YYYY-MM-DD_HH-MM-SS}``.
    ``name`` and ``description`` come from the ``logging`` config section.
    The folder holds ``smartsight.log`` and ``config.yaml``, a copy of the
    effective config with secret fields redacted.

    A missing ``logging`` section, or ``file_enabled`` false, creates nothing.
    A second call keeps the folder from the first call. ``logs_root`` defaults
    to ``<repo>/logs`` and exists so tests can point at a temporary directory.
    """
    if not isinstance(config, dict):
        return None
    section = config.get("logging")
    if not isinstance(section, dict) or not section.get("file_enabled", False):
        return None

    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARK, False):
            return getattr(handler, "run_directory", None)

    name = _path_piece(section.get("name") or "smartsight", fallback="smartsight")
    description = _path_piece(section.get("description") or "run", fallback="run")
    moment = started_at or datetime.now()
    stamp = moment.strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = Path(logs_root) if logs_root is not None else _REPO_ROOT / "logs"
    folder = root_dir / f"{name}_{description}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    _write_config_copy(folder / "config.yaml", config)

    handler = logging.FileHandler(folder / "smartsight.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, _HANDLER_MARK, True)
    handler.run_directory = folder
    root.addHandler(handler)
    get_logger("logging_setup").info(f"Writing logs to {folder}")
    return folder


def _path_piece(value, fallback):
    text = _UNSAFE_PATH.sub("", str(value)).strip().strip(".")
    text = re.sub(r"\s+", "_", text)
    text = text[:80]
    return text or fallback


def _redact(value):
    """Return a copy of config data with secret-like fields replaced."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if isinstance(key, str) and _SECRET_KEY.search(key):
                cleaned[key] = "***"
            else:
                cleaned[key] = _redact(item)
        return cleaned
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _write_config_copy(path, config):
    body = _dump_yaml(_redact(config))
    path.write_text(
        "# Effective config for this run. Secret fields are redacted.\n" + body,
        encoding="utf-8",
    )


def _plain_key(value):
    return isinstance(value, str) and value.isidentifier()


def _yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def _dump_yaml(value, indent=0):
    """Write the config shapes SmartSight actually stores."""
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return f"{pad}{{}}\n"
        chunks = []
        for key, item in value.items():
            key_text = str(key) if _plain_key(key) else _yaml_scalar(key)
            if isinstance(item, (dict, list)) and item:
                chunks.append(f"{pad}{key_text}:\n{_dump_yaml(item, indent + 1)}")
            else:
                chunks.append(f"{pad}{key_text}: {_yaml_scalar(item)}\n")
        return "".join(chunks)
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                chunks.append(f"{pad}-\n{_dump_yaml(item, indent + 1)}")
            else:
                chunks.append(f"{pad}- {_yaml_scalar(item)}\n")
        return "".join(chunks)
    return f"{pad}{_yaml_scalar(value)}\n"
