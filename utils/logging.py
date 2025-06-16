import logging


def setup_logging(level=logging.INFO, log_to_file=False, filename="app.log") -> None:
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(filename))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
