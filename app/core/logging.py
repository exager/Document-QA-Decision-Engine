import logging
import json
from datetime import datetime
from contextvars import ContextVar

request_id_ctx = ContextVar("request_id", default=None)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None) or request_id_ctx.get()
        if request_id:
            payload["request_id"] = request_id

         # Include all structured extras
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in (
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info",
                "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "taskName",
            ):
                continue
            if key not in payload:
                payload[key] = value
        return json.dumps(payload)


def setup_logging(log_level: str = "INFO"):
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers = [handler]
