import logging


class Logger:
    def __init__(self, level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger("game_master")

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)
