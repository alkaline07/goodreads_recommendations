"""
Logging Configuration Module for Goodreads Data Pipeline

This module provides centralized logging configuration for all pipeline components.
It creates persistent log files that survive Airflow reruns and provides both
file and console logging with consistent formatting.

Key Features:
- JSON formatted logs for ELK stack integration
- Creates persistent log files in the logs/ directory
- Avoids duplicate handlers when DAGs are re-imported
- Provides both file and console logging (console disabled in Airflow)
- Supports extra fields for searchable logs in Kibana

Author: Goodreads Recommendation Team
Date: 2025
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Formats logs as JSON for ELK stack integration"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_fields") and record.extra_fields:
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class PlainFormatter(logging.Formatter):
    """Plain text formatter for file logs"""

    def __init__(self):
        super().__init__('%(asctime)s [%(levelname)s] %(name)s: %(message)s')


class ELKLogger(logging.Logger):
    """Logger that supports extra keyword arguments for ELK searchability"""

    def _log_with_extras(self, level: int, msg: str, exc_info=None, **kwargs: Any) -> None:
        extra = {"extra_fields": kwargs} if kwargs else {}
        super().log(level, msg, exc_info=exc_info, extra=extra)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log_with_extras(logging.INFO, msg, **kwargs)

    def error(self, msg: str, exc_info=None, **kwargs: Any) -> None:
        self._log_with_extras(logging.ERROR, msg, exc_info=exc_info, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log_with_extras(logging.WARNING, msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log_with_extras(logging.DEBUG, msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._log_with_extras(logging.ERROR, msg, exc_info=True, **kwargs)


_loggers: dict = {}


def get_logger(name: str, use_json: bool = True) -> ELKLogger:
    """
    Create and configure a logger for the specified component.

    Args:
        name: Name of the logger (typically the module or component name)
        use_json: If True, output JSON format (for ELK). If False, plain text.

    Returns:
        ELKLogger: Configured logger instance with extra fields support

    Examples:
        # Basic usage
        logger = get_logger("data-cleaning")
        logger.info("Processing started")
        logger.error("Failed to process")

        # With extra fields (searchable in Kibana)
        logger.info("User action", user_id="123", action="login")
        logger.error("Database error", error="timeout", query="SELECT *")
    """
    if name in _loggers:
        return _loggers[name]

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    logging.setLoggerClass(ELKLogger)
    logger = logging.getLogger(name)
    logging.setLoggerClass(logging.Logger)

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(PlainFormatter())
        logger.addHandler(file_handler)

        if 'AIRFLOW_HOME' not in os.environ:
            console_handler = logging.StreamHandler(sys.stdout)
            if use_json:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(PlainFormatter())
            logger.addHandler(console_handler)
        else:
            json_handler = logging.StreamHandler(sys.stdout)
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)

    logger.propagate = False
    _loggers[name] = logger
    return logger
