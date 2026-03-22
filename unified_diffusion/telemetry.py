from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "unified_diffusion"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        stream_handler = logger.handlers[0]
        if (
            isinstance(stream_handler, logging.StreamHandler)
            and getattr(stream_handler, "stream", None) is not None
            and stream_handler.stream is sys.stderr
            and not getattr(stream_handler.stream, "closed", False)
        ):
            return logger
        logger.handlers.clear()

    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_event(event: str, **fields: object) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    get_logger().info(json.dumps(payload, ensure_ascii=True, sort_keys=True))
