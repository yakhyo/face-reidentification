import logging


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = False,
    filename: str = "app.log",
) -> None:
    """Configure root logger with console (and optional file) output.

    Args:
        level: Logging level (e.g. ``logging.INFO``).
        log_to_file: Whether to also write logs to a file.
        filename: Log file path (used only when *log_to_file* is True).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(filename, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
