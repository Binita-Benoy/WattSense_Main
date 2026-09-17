import logging
import os
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime

from config import OUT_LOGS

os.makedirs(OUT_LOGS, exist_ok=True)


def configure_logging(name: str):
    """Configure one file-only log for the entire pipeline run."""
    log_path = os.path.join(
        OUT_LOGS,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove previous handlers so every message goes to this run's log only.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger, log_path


@contextmanager
def capture_terminal_output(log_path):
    """Redirect remaining stdout/stderr (e.g. TensorFlow) into the same log."""
    with open(log_path, "a", encoding="utf-8", buffering=1) as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            yield
