import logging
from pathlib import Path


def create_file_logger(
    name: str,
    file_path: str,
) -> logging.Logger:
    """
    Creates a logger that writes messages to a specific file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    path = Path(file_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handler = logging.FileHandler(
        path,
        mode="w",
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

    logger.addHandler(handler)

    return logger


files_logger = create_file_logger(
    "files",
    "data/logs/files_read.log",
)

chunks_logger = create_file_logger(
    "chunks",
    "data/logs/chunks.log",
)

chunks_loaded_logger = create_file_logger(
    "chunks_loaded",
    "data/logs/chunks_loaded.log",
)

steps_logger = create_file_logger(
    "steps",
    "data/logs/steps.log",
)
