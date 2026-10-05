import logging
import os
import sys
from datetime import datetime

from config import OUT_LOGS

os.makedirs(OUT_LOGS, exist_ok=True)


def configure_logging(name: str):
    """Send pipeline messages to both the terminal and a timestamped log file."""
    log_path = os.path.join(
        OUT_LOGS,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_path


def log_print(message="", level=logging.INFO):
    """Print to terminal and write the same message to the active log file."""
    print(message, flush=True)
    logging.getLogger().log(level, message)


def safe_savefig(fig, path, dpi=150, bbox_inches="tight", **kwargs):
    """Save a Matplotlib figure without allowing a filesystem/Windows lock to abort the ML pipeline.

    The requested path is tried first.  If Windows rejects that specific filename
    (for example because an existing PNG is locked by another application), a
    timestamped filename in the same directory is tried, followed by the current
    working directory.  If all locations fail, the error is logged and execution
    continues because graph export is evidence/output, not part of model scoring.
    """
    from pathlib import Path

    target = Path(str(path)).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logging.getLogger().warning(
            "Could not create graph directory %s: %s", target.parent, exc
        )

    candidates = [
        target,
        target.with_name(f"{target.stem}_{timestamp}{target.suffix}"),
        Path.cwd() / f"{target.stem}_{timestamp}{target.suffix}",
    ]

    errors = []
    for candidate in candidates:
        try:
            # An existing directory with a .png name cannot be overwritten as a file.
            if candidate.exists() and candidate.is_dir():
                raise IsADirectoryError(f"Target is a directory: {candidate}")

            candidate.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(candidate),
                dpi=dpi,
                bbox_inches=bbox_inches,
                **kwargs,
            )

            if candidate != target:
                logging.getLogger().warning(
                    "Could not use requested graph path %s; graph saved to fallback %s",
                    target, candidate,
                )
            else:
                logging.getLogger().info("Graph saved: %s", candidate)
            return str(candidate)
        except (OSError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")

    logging.getLogger().warning(
        "Graph could not be saved, but the ML pipeline will continue. Attempts: %s",
        " | ".join(errors),
    )
    return None
