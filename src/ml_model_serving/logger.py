import logging


def setup(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
